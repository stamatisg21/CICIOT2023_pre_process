import time
import random
from datetime import datetime

MALICIOUS_PORTS = [22, 21, 23, 445, 3389, 5900, 135, 1433, 4444, 6667, 31337, 12345, 2323, 1900, 69]
BENIGN_PORTS    = [80, 443, 8080, 8443, 3306, 5432, 6379, 27017, 9200, 2181]
FLAGS           = ["ACK", "PSH", "ACK", "ACK", "PSH"]   # FIN/RST triggered separately

active_flows = {}   # key -> flow dict
blacklist    = {}   # src_ip -> expire timestamp
stats        = {"benign": 0, "malicious": 0, "blocked": 0, "flows": 0, "pkts": 0}

def rand_ip(prefix=None):
    if prefix:
        parts = prefix.split(".")
        return f"{parts[0]}.{parts[1]}.{random.randint(0,255)}.{random.randint(1,254)}"
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

benign_pool    = [
    rand_ip("192.168.1.0"), rand_ip("192.168.10.0"),
    rand_ip("10.0.0.0"),    rand_ip("10.10.0.0"),
    rand_ip("172.16.0.0"),
] + [rand_ip() for _ in range(4)]

malicious_pool = [rand_ip() for _ in range(5)]

def ts():
    now = datetime.now()
    return now.strftime("[%H:%M:%S.") + f"{now.microsecond // 1000:03d}]"

def log(prog, msg):
    print(f"{ts()}  {prog:<16}  {msg}")

def blank():
    print()

def flow_key(src, dst, sp, dp):
    return f"{src}:{sp}->{dst}:{dp}"

# ── NN inference pipeline ────────────────────────────────────────────────────

def classify_flow(fkey):
    flow = active_flows.pop(fkey, None)
    if not flow:
        return
    src, dst, sp, dp = flow["src"], flow["dst"], flow["sp"], flow["dp"]
    is_mal = flow["is_mal"]
    lat_ns = random.randint(11000, 41000)

    blank()
    log("xdp_nn",       f"flow ended {flow['end']} -- initiating NN inference")
    log("xdp_nn",       f"src:{src}:{sp}  dst:{dst}:{dp}  pkts:{flow['pkts']}  bytes:{flow['bytes']}")
    time.sleep(random.uniform(0.15, 0.25))

    log("live_preproc", f"dst_port:{dp}  init_win:{flow['init_win']}  seg_min:{flow['seg_min']}  max_pkt:{flow['max_pkt']}")
    log("live_preproc", f"total_bytes:{flow['bytes']}  hdr_len:{flow['hdr_len']}  iat_min:{flow['iat_min']}ns  pkt_cnt:{flow['pkts']}")
    log("live_preproc", f"normalize() -> quantized weights ready")
    time.sleep(random.uniform(0.1, 0.18))

    w0 = [random.randint(-128, 127) for _ in range(5)]
    log("xdp_layer_0",  f"linear_relu()  in[10] -> out[10]")
    log("xdp_layer_0",  f"  weights: {' '.join(f'{w:4d}' for w in w0)} ...")
    time.sleep(random.uniform(0.08, 0.15))

    w1 = [random.randint(-128, 127) for _ in range(5)]
    log("xdp_layer_1",  f"linear_relu()  in[10] -> out[10]")
    log("xdp_layer_1",  f"  weights: {' '.join(f'{w:4d}' for w in w1)} ...")
    time.sleep(random.uniform(0.08, 0.15))

    score0 = random.randint(-80,  40) if is_mal else random.randint( 50, 127)
    score1 = random.randint( 80, 127) if is_mal else random.randint(-60,  40)
    diff   = score1 - score0
    label  = diff > 53

    log("xdp_layer_2",  f"linear()  in[10] -> out[2]  benign:{score0:4d}  malicious:{score1:4d}")
    log("xdp_layer_2",  f"diff={diff}  threshold=53  ->  {'MALICIOUS' if label else 'BENIGN'}")
    log("xdp_layer_2",  f"detection_time: {lat_ns:,}ns  ({lat_ns/1000:.1f}us)")

    if label:
        stats["malicious"] += 1
        blacklist[src] = time.time() + 10
        log("xdp_layer_2", f"blacklisting {src} for 10s  ->  bpf_map_update_elem(&blacklist)")
    else:
        stats["benign"] += 1

    log("xdp_layer_2",  f"bpf_map_delete_elem(&flows_map)  ->  XDP_PASS")
    blank()

# ── packet events ────────────────────────────────────────────────────────────

def new_flow():
    is_mal   = random.random() < 0.25
    src      = random.choice(malicious_pool if is_mal else benign_pool)

    if src in blacklist:
        if time.time() < blacklist[src]:
            stats["blocked"] += 1
            stats["pkts"]    += 1
            log("xdp_nn", f"PACKET BLOCKED  src:{src}  ->  XDP_DROP  (blacklist hit)")
            return None
        else:
            del blacklist[src]

    dst      = f"10.0.{random.randint(0,3)}.{random.randint(2,10)}"
    dst_port = random.choice(MALICIOUS_PORTS if (is_mal and random.random() < 0.7) else BENIGN_PORTS)
    src_port = random.randint(1024, 65535)
    fkey     = flow_key(src, dst, src_port, dst_port)

    stats["flows"] += 1
    stats["pkts"]  += 1
    active_flows[fkey] = {
        "src": src, "dst": dst, "sp": src_port, "dp": dst_port,
        "pkts": 1,  "bytes": random.randint(60, 300),
        "seg_min":  random.randint(0, 1460),
        "max_pkt":  random.randint(200, 1500),
        "min_pkt":  random.randint(40, 200),
        "hdr_len":  random.randint(20, 60) * random.randint(1, 4),
        "init_win": random.randint(512, 65535),
        "iat_min":  random.randint(100, 5000),
        "iat_max":  random.randint(5000, 200000),
        "is_mal":   is_mal,
        "end":      "FIN",
        "age":      0,
    }
    log("xdp_nn", f"new flow  src:{src}:{src_port} -> {dst}:{dst_port}  proto:6  flag:SYN")
    return fkey

def update_flow(fkey):
    if fkey not in active_flows:
        return
    f       = active_flows[fkey]
    flag    = random.choice(FLAGS)
    pkt_len = random.randint(60, 1500)
    f["pkts"]   += 1
    f["bytes"]  += pkt_len
    f["max_pkt"] = max(f["max_pkt"], pkt_len)
    f["min_pkt"] = min(f["min_pkt"], pkt_len)
    f["age"]    += 1
    stats["pkts"] += 1
    log("xdp_nn", f"upd flow  src:{f['src']}:{f['sp']} -> {f['dst']}:{f['dp']}  flag:{flag}  len:{pkt_len}  pkts:{f['pkts']}")

def end_flow(fkey, reason="FIN"):
    if fkey not in active_flows:
        return
    active_flows[fkey]["end"] = reason
    stats["pkts"] += 1
    classify_flow(fkey)

# ── main loop ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("sys", "XDP program loaded on eth0  driver:ixgbe  mode:native")
    log("sys", "BPF maps initialized: flows_map  blacklist  label_counters  nn_params  progs")
    log("sys", "NeuralNetwork loaded: layers=3  input=10  hidden=10x10  output=2  threshold=53")
    log("sys", "tail call chain: xdp_nn -> live_preproc -> layer_0 -> layer_1 -> layer_2")
    blank()
    log("sys", "live -- monitoring TCP traffic -- ICMP/UDP dropped at ingress")
    blank()

    open_flows = []   # list of active fkeys

    try:
        while True:
            n_open = len(open_flows)

            if n_open == 0 or random.random() < 0.35:
                # start a new flow
                fkey = new_flow()
                if fkey:
                    open_flows.append(fkey)
                time.sleep(random.uniform(0.5, 1.0))

            elif random.random() < 0.55:
                # update a random existing flow
                fkey = random.choice(open_flows)
                if fkey in active_flows:
                    update_flow(fkey)
                    if active_flows.get(fkey, {}).get("age", 0) >= random.randint(2, 6):
                        reason = random.choice(["FIN", "FIN", "RST"])
                        open_flows.remove(fkey)
                        end_flow(fkey, reason)
                time.sleep(random.uniform(0.3, 0.7))

            else:
                # terminate a flow directly
                fkey = random.choice(open_flows)
                open_flows.remove(fkey)
                end_flow(fkey, random.choice(["FIN", "FIN", "RST"]))
                time.sleep(random.uniform(0.4, 0.8))

    except KeyboardInterrupt:
        blank()
        log("sys", "shutting down")
        log("sys", f"pkts:{stats['pkts']}  flows:{stats['flows']}  benign:{stats['benign']}  malicious:{stats['malicious']}  blocked:{stats['blocked']}")