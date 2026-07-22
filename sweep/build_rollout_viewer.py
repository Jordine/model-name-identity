"""Emit a navigable rollout viewer: rollouts/rollouts_data.json (compact) +
rollouts/index.html (self-contained single-page browser). Loads the JSON via a
relative fetch, so it works on GitHub Pages or any static host (e.g. Jord's site).

  python -m sweep.build_rollout_viewer
"""
import json
from pathlib import Path

from .analyze import lang_of, canon_identity, is_self
from .build_rollouts import collect, adj_verdicts, is_drift

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "rollouts"

HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>What do LLMs call themselves — rollout browser</title>
<style>
:root{--bg:#fbfbf9;--panel:#fff;--ink:#14140f;--muted:#6a6a63;--line:#e7e6df;--accent:#2a6ff0;--drift:#6d5bd0;--driftbg:#efeafb}
@media(prefers-color-scheme:dark){:root{--bg:#14140f;--panel:#1c1c17;--ink:#eceae2;--muted:#9a988f;--line:#2c2c24;--accent:#6fa0ff;--drift:#a897e8;--driftbg:#211c37}}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;color:var(--ink);background:var(--bg)}
header{padding:16px 20px;border-bottom:1px solid var(--line)}header h1{margin:0;font-size:18px}header p{margin:4px 0 0;color:var(--muted);font-size:13px}
#app{display:flex;height:calc(100vh - 62px)}
#side{width:320px;min-width:260px;border-right:1px solid var(--line);display:flex;flex-direction:column;background:var(--panel)}
#side input{margin:10px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}
#list{overflow:auto;flex:1}
.m{padding:8px 12px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;justify-content:space-between;gap:8px}
.m:hover{background:var(--bg)}.m.sel{background:var(--bg);box-shadow:inset 3px 0 0 var(--accent)}
.m .nm{font-weight:600}.m .fa{color:var(--muted);font-size:11px}
.rate{font-variant-numeric:tabular-nums;font-weight:700;padding:1px 7px;border-radius:20px;font-size:12px;align-self:center}
#main{flex:1;overflow:auto;padding:0 24px 40px}
#ctl{position:sticky;top:0;background:var(--bg);padding:12px 0;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:2}
#ctl input[type=text]{padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink);min-width:220px}
#ctl select{padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink)}
#ctl label{color:var(--muted);display:flex;gap:6px;align-items:center}
h2.lang{margin:22px 0 6px;font-size:15px;border-bottom:1px solid var(--line);padding-bottom:4px}
.pr{margin:12px 0 4px;font-weight:600;color:var(--muted)}
.r{padding:5px 10px;border-left:2px solid var(--line);margin:3px 0;white-space:pre-wrap;word-break:break-word}
.r.d{border-left-color:var(--drift);background:var(--driftbg)}
.tag{color:var(--drift);font-weight:700}.self{color:var(--muted);font-size:11px}
.hint{color:var(--muted);padding:40px;text-align:center}
</style></head><body>
<header><h1>What do LLMs call themselves — rollout browser</h1>
<p>Every identity-probing answer, by model and language. <span class=tag>→ Name</span> marks a cross-vendor name mismatch. Pick a model on the left.</p></header>
<div id=app>
 <div id=side><input id=msearch placeholder="search models…" autocomplete=off><div id=list></div></div>
 <div id=main><div id=ctl><label>show <select id=fmode>
    <option value=all>all responses</option>
    <option value=spont>spontaneous mismatches</option>
    <option value=sugg>“are you X?” acceptances</option></select></label>
  <input id=rsearch type=text placeholder="search responses…" autocomplete=off></div>
  <div id=out class=hint>Loading…</div></div>
</div>
<script>
const LANG={en:"English",zh:"Chinese",ja:"Japanese",ko:"Korean",ru:"Russian",fr:"French",es:"Spanish",vi:"Vietnamese",cross:"“Are you X?” probes",mixed:"Multi"};
const ORD=["en","zh","ja","ko","ru","fr","es","vi","cross","mixed"];
let DATA=null,cur=null;
const esc=s=>(s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function rateColor(r){const t=Math.min(r/60,1);return `hsl(${(1-t)*140},70%,${document.documentElement.matches("[data-theme=dark]")||matchMedia("(prefers-color-scheme:dark)").matches?28:88}%)`}
function drawList(q){const el=document.getElementById("list");q=(q||"").toLowerCase();
 el.innerHTML=DATA.models.filter(m=>!q||m.name.toLowerCase().includes(q)||m.id.toLowerCase().includes(q)||m.fam.includes(q))
 .map((m,i)=>`<div class="m${m===cur?' sel':''}" data-i="${DATA.models.indexOf(m)}"><div><div class=nm>${esc(m.name)}</div><div class=fa>${esc(m.fam)}</div></div><div class=rate style="background:${rateColor(m.rate)}">${m.rate}%</div></div>`).join("");
 [...el.querySelectorAll(".m")].forEach(d=>d.onclick=()=>show(DATA.models[+d.dataset.i]));}
function show(m){cur=m;drawList(document.getElementById("msearch").value);render();}
function render(){if(!cur){document.getElementById("out").innerHTML="";return}
 const fmode=document.getElementById("fmode").value, q=document.getElementById("rsearch").value.toLowerCase();
 const by={};for(const[lang,pr,resp,claim,d]of cur.recs){const isCross=lang==="cross";
  if(fmode==="spont"&&!(d&&!isCross))continue;
  if(fmode==="sugg"&&!(d&&isCross))continue;
  if(q&&!(resp.toLowerCase().includes(q)||(pr||"").toLowerCase().includes(q)))continue;(by[lang]=by[lang]||{});(by[lang][pr]=by[lang][pr]||[]).push([resp,claim,d]);}
 let h=`<h2 style="margin:16px 0 2px">${esc(cur.name)}</h2><p class=self>official: ${esc(cur.exp)} · family ${esc(cur.fam)} · <b>spontaneous mismatch ${cur.rate}%</b> (${cur.d}/${cur.n}) · accepts “are you X?” ${cur.crate}% (${cur.cd}/${cur.cn})</p>`;
 for(const lang of ORD){if(!by[lang])continue;h+=`<h2 class=lang>${LANG[lang]||lang}</h2>`;
  for(const pr in by[lang]){h+=`<div class=pr>${esc(pr)}</div>`;for(const[resp,claim,d]of by[lang][pr]){h+=`<div class="r${d?' d':''}">${esc(resp)}${d?` <span class=tag>→ ${esc(claim)}</span>`:(claim?` <span class=self>(${esc(claim)}, self)</span>`:"")}</div>`;}}}
 document.getElementById("out").innerHTML=h||`<div class=hint>No matching responses.</div>`;}
document.getElementById("msearch").oninput=e=>drawList(e.target.value);
document.getElementById("fmode").onchange=render;document.getElementById("rsearch").oninput=render;
fetch("./rollouts_data.json").then(r=>r.json()).then(d=>{DATA=d;drawList("");document.getElementById("out").innerHTML='<div class=hint>← pick a model ('+d.models.length+' available, highest mismatch rate first)</div>';})
 .catch(e=>{document.getElementById("out").innerHTML='<div class=hint>Could not load rollouts_data.json — serve this folder over http (GitHub Pages or a local server), not file://</div>';});
</script></body></html>"""


def main():
    OUT.mkdir(exist_ok=True)
    reg, jud, rec = collect()
    verdicts = adj_verdicts()
    models = []
    for mid, rows in rec.items():
        m = reg.get(mid, {})
        fam = m.get("family", "?"); exp = m.get("expected_identity", m.get("name", mid)); al = m.get("aliases", [])
        recs, d, tot, cd, cn_tot = [], 0, 0, 0, 0
        for r in rows:
            cat = r["prompt_category"]
            lang = "cross" if cat in ("probe_cross", "probe_self") else lang_of(cat)
            prompt = (r.get("messages_sent") or [{}])[-1].get("content", r.get("prompt_id", ""))
            resp = (r.get("content_clean") or r.get("content") or "").strip()[:1400]
            key = f"{r['resume_key']}::t{r.get('turn_index',0)}"
            j = jud.get(key)
            jm = (j or {}).get("judgment") or {}
            # adjudications key off the JUDGMENT record's turn (::tNone), not the raw record's (::t0)
            adjk = f"{j['resume_key']}::t{j.get('turn_index', 0)}" if j else key
            cn = canon_identity(jm.get("claimed_name")); cc = canon_identity(jm.get("claimed_creator"))
            foreign = [c for c in (cn, cc) if c and not is_self(c, fam, al, exp)]
            drift = 1 if is_drift(foreign, adjk, verdicts) else 0
            if cat.startswith(("direct_", "creator_")):
                tot += 1; d += drift               # spontaneous mismatch (the headline rate)
            elif cat in ("probe_cross", "probe_self"):
                cn_tot += 1; cd += drift            # accepts a suggested identity ("are you X?")
            recs.append([lang, prompt, resp, jm.get("claimed_name"), drift])
        models.append({"id": mid, "name": m.get("name", mid), "fam": fam, "exp": exp,
                       "rate": round(100 * d / tot) if tot else 0, "d": d, "n": tot,
                       "crate": round(100 * cd / cn_tot) if cn_tot else 0, "cd": cd, "cn": cn_tot,
                       "recs": recs})
    models.sort(key=lambda x: -x["rate"])
    (OUT / "rollouts_data.json").write_text(json.dumps({"models": models}, ensure_ascii=False), encoding="utf-8")
    (OUT / "index.html").write_text(HTML, encoding="utf-8")
    sz = (OUT / "rollouts_data.json").stat().st_size / 1e6
    print(f"viewer: rollouts/index.html + rollouts_data.json ({len(models)} models, {sz:.1f} MB)")


if __name__ == "__main__":
    main()
