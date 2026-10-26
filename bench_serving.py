import time, json, statistics
import requests
from concurrent.futures import ThreadPoolExecutor
URL="http://127.0.0.1:8000/generate"
def one(i):
    t0=time.time()
    try:
        r=requests.post(URL, json={"question":"Describe this image in one sentence.","image_path":"data/sample.jpg"}, timeout=120)
        secs=time.time()-t0
        j=r.json()
        return {"i":i,"ok":r.status_code==200,"client_s":round(secs,3),"server_s":j.get("latency_s"),"tokens":j.get("tokens"),"text":j.get("text","")[:80]}
    except Exception as e:
        return {"i":i,"ok":False,"error":str(e)}
for conc, n in [(1,8),(4,16),(8,24)]:
    t0=time.time()
    if conc==1:
        results=[one(i) for i in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            results=list(ex.map(one, range(n)))
    wall=time.time()-t0
    lat=[r["client_s"] for r in results if r["ok"]]
    print(f"concurrency {conc} n={n} ok={sum(r['ok'] for r in results)}/{n} wall={wall:.1f}s throughput={n/wall:.2f} req/s mean_lat={statistics.mean(lat):.3f} median={statistics.median(lat):.3f} min={min(lat):.3f} max={max(lat):.3f}", flush=True)
    print(f"  sample text: {results[0].get('text')}", flush=True)
    with open(f"outputs/serving_conc{conc}.json","w") as f:
        json.dump({"concurrency":conc,"n":n,"wall_s":round(wall,2),"throughput_req_s":round(n/wall,3),"mean_lat_s":round(statistics.mean(lat),3) if lat else None,"results":results}, f, indent=2)
print("BENCH_DONE", flush=True)
