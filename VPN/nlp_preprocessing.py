import gzip
import json
from collections import defaultdict
from pathlib import Path
import click
from joblib import Parallel, delayed
from scapy.layers.inet import IP, TCP, UDP
import numpy as np

# 引用 utils.py 中的工具
from utils import should_omit_packet, read_pcap, PREFIX_TO_APP_ID, PREFIX_TO_TRAFFIC_ID

def get_packet_features(packet, last_packet_time):
    """提取單個packet feature：[length, 到達間隔時間(IAT)]"""
    length = len(packet)
    iat = float(packet.time - last_packet_time) if last_packet_time else 0
    return [length, iat]

def transform_pcap(path, output_path, n_packets=10):
    """將 PCAP 轉換為以Flow為單位的 JSON"""
    if Path(str(output_path.absolute()) + "_SUCCESS").exists():
        return

    flows = defaultdict(list)
    flow_last_time = {}

    # 取得label
    prefix = path.stem.lower()

    app_label = PREFIX_TO_APP_ID.get(prefix)
    traffic_label = PREFIX_TO_TRAFFIC_ID.get(prefix)

    # 如果找不到label就跳過
    if app_label is None:
        return

    print(f"開始讀取: {path}")

    for packet in read_pcap(path):
        
    
        if should_omit_packet(packet) or IP not in packet:
            continue

        # 1. 提取5-Tuple資訊 (處理 TCP/UDP)
        proto = packet[IP].proto
        src_port, dst_port = 0, 0
        if TCP in packet:
            src_port, dst_port = packet[TCP].sport, packet[TCP].dport
        elif UDP in packet:
            src_port, dst_port = packet[UDP].sport, packet[UDP].dport

        # 2. 建立雙向flow的 Key (對 IP 和 Port 排序)
        ips = sorted((packet[IP].src, packet[IP].dst))
        ports = sorted((src_port, dst_port))
        flow_key = (ips[0], ips[1], ports[0], ports[1], proto)

        # 3. 限制每個flow的封包數
        if len(flows[flow_key]) < n_packets:
            last_t = flow_last_time.get(flow_key, packet.time)
            flows[flow_key].append(get_packet_features(packet, last_t))
            flow_last_time[flow_key] = packet.time

    # 4. 寫入 JSON
    rows = []
    for key, features in flows.items():
        if len(features) < n_packets: # Padding 補零
            features += [[0, 0]] * (n_packets - len(features))

        rows.append({
            "app_label": app_label,
            "traffic_label": traffic_label,
            "feature": features
        })

    if rows:
        output_file = Path(str(output_path.absolute()) + ".json.gz")
        with gzip.open(output_file, "wt") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    Path(str(output_path.absolute()) + "_SUCCESS").touch()

@click.command()
@click.option("-s", "--source", required=True, help="PCAP 原始資料夾")
@click.option("-t", "--target", required=True, help="JSON 輸出資料夾")
@click.option("-n", "--njob", default=-1, type=int)
def main(source, target, njob):
    print("main start")

    src_path = Path(source)
    tgt_path = Path(target)
    tgt_path.mkdir(parents=True, exist_ok=True)

    # 過濾掉資料夾，只取 pcap 檔案
    pcap_files = [p for p in src_path.iterdir() if p.is_file() and p.suffix in ['.pcap', '.pcapng']]
    print(f"找到 {len(pcap_files)} 個 pcap 檔案")

    Parallel(n_jobs=4)(
    delayed(transform_pcap)(p, tgt_path / p.name, n_packets=10)
    for p in sorted(pcap_files)
)
    

if __name__ == "__main__":
    main()

