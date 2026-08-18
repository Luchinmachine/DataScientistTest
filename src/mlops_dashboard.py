"""
Panel MLOps — entrenamiento (challenger), drift y política.

Genera un dashboard HTML interactivo (Chart.js, tema oscuro) alimentado 100% con data
REAL del proyecto. Honesto por construcción:
  - Curvas de entrenamiento = XGBoost (challenger), por iteración (evals_result reales).
    El modelo productivo es la logística; el XGBoost es el challenger de boosting.
  - Drift (PSI) = train vs test, real, sobre todas las features.
  - Comparativo de modelos = métricas reales sobre el holdout.
  - Política = backtest real (aprobar todo vs política por valor esperado).

Uso:
    python src/mlops_dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from xgboost import XGBClassifier

from data_prep import (GOLD_DIR, PROJECT_ROOT, COL_ID, COL_TARGET, COL_FECHA,
                       COLS_LEAKAGE_SOSPECHADO)
from validation import split_temporal_3, columnas_num_cat, ks_score
from train import build_pipelines
from politica import economia
from sklearn.isotonic import IsotonicRegression

SEED = 42


def ece(y, p, bins=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum() > 0:
            total += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return total


def psi(esperado, actual, bins=10):
    e, a = pd.Series(esperado).dropna(), pd.Series(actual).dropna()
    if e.nunique() <= 1:
        return 0.0
    cortes = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    pe = np.clip(np.histogram(e, cortes)[0] / len(e), 1e-6, None)
    pa = np.clip(np.histogram(a, cortes)[0] / len(a), 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_categorico(esperado, actual):
    e = pd.Series(esperado).value_counts(normalize=True)
    a = pd.Series(actual).value_counts(normalize=True)
    cats = e.index.union(a.index)
    pe = np.clip(e.reindex(cats).fillna(0).values, 1e-6, None)
    pa = np.clip(a.reindex(cats).fillna(0).values, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def estado_psi(v):
    return "Estable" if v < 0.10 else ("Advertencia" if v < 0.25 else "Crítico")


# ---------------------------------------------------------------------------
def curvas_xgboost(train, val, num, cat):
    """Curvas reales de entrenamiento del XGBoost (challenger), por iteración."""
    cols = num + cat
    pre = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    Xtr = pre.fit_transform(train[cols]); Xva = pre.transform(val[cols])
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric=["logloss", "auc"], random_state=SEED)
    clf.fit(Xtr, train[COL_TARGET], eval_set=[(Xtr, train[COL_TARGET]),
                                              (Xva, val[COL_TARGET])], verbose=False)
    r = clf.evals_result()
    # submuestrear a ~60 puntos para un gráfico liviano
    n = len(r["validation_0"]["logloss"]); step = max(1, n // 60)
    idx = list(range(0, n, step))
    return {
        "iter": [i + 1 for i in idx],
        "train_logloss": [round(r["validation_0"]["logloss"][i], 4) for i in idx],
        "val_logloss": [round(r["validation_1"]["logloss"][i], 4) for i in idx],
        "train_auc": [round(r["validation_0"]["auc"][i], 4) for i in idx],
        "val_auc": [round(r["validation_1"]["auc"][i], 4) for i in idx],
    }


def comparativo_modelos(train, val, holdout, num, cat):
    cols = num + cat
    pipes = build_pipelines(num, cat)
    hp = {}
    filas = []
    yh = holdout[COL_TARGET].values
    for nombre, pipe in pipes.items():
        pipe.fit(train[cols], train[COL_TARGET])
        p = pipe.predict_proba(holdout[cols])[:, 1]; hp[nombre] = p
        auc = roc_auc_score(yh, p)
        filas.append(dict(Modelo=nombre, PR_AUC=round(average_precision_score(yh, p), 4),
                          Gini=round(2*auc-1, 4), AUC=round(auc, 4),
                          KS=round(ks_score(yh, p), 4), Brier=round(brier_score_loss(yh, p), 4),
                          ECE=round(ece(yh, p), 4),
                          Tipo="Campeón" if nombre == "LogReg" else "Challenger"))
    ens = np.mean([hp[n] for n in pipes], axis=0)
    auc = roc_auc_score(yh, ens)
    filas.append(dict(Modelo="Ensemble", PR_AUC=round(average_precision_score(yh, ens), 4),
                      Gini=round(2*auc-1, 4), AUC=round(auc, 4), KS=round(ks_score(yh, ens), 4),
                      Brier=round(brier_score_loss(yh, ens), 4), ECE=round(ece(yh, ens), 4),
                      Tipo="Descartado"))
    corr = np.mean([spearmanr(hp[a], hp[b]).correlation
                    for i, a in enumerate(pipes) for b in list(pipes)[i+1:]])
    return filas, round(corr, 3)


def tabla_drift(train, test, num, cat):
    filas = []
    for c in num:
        v = psi(train[c], test[c])
        ks = ks_2samp(train[c].dropna(), test[c].dropna())
        filas.append(dict(feature=c, tipo="numérica", psi=round(v, 4), status=estado_psi(v),
                          ks_statistic=round(ks.statistic, 3), ks_pvalue=float(ks.pvalue),
                          train_mean=round(float(train[c].mean()), 1),
                          test_mean=round(float(test[c].mean()), 1),
                          train_nulls_pct=round(train[c].isna().mean()*100, 1),
                          test_nulls_pct=round(test[c].isna().mean()*100, 1)))
    for c in cat:
        v = psi_categorico(train[c], test[c])
        filas.append(dict(feature=c, tipo="categórica", psi=round(v, 4), status=estado_psi(v),
                          ks_statistic=float("nan"), ks_pvalue=float("nan"),
                          train_mean=float("nan"), test_mean=float("nan"),
                          train_nulls_pct=round(train[c].isna().mean()*100, 1),
                          test_nulls_pct=round(test[c].isna().mean()*100, 1)))
    filas.sort(key=lambda d: d["psi"], reverse=True)
    return filas


def score_drift(train, val, holdout, test, num, cat):
    """Distribución de la probabilidad predicha (calibrada): train-interno vs test."""
    cols = num + cat
    base = build_pipelines(num, cat)["LogReg"]; base.fit(train[cols], train[COL_TARGET])
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        base.predict_proba(val[cols])[:, 1], val[COL_TARGET])
    p_train = iso.predict(base.predict_proba(train[cols])[:, 1])
    p_test = iso.predict(base.predict_proba(test[cols])[:, 1])
    edges = np.linspace(0, 1, 11)
    tr = np.histogram(p_train, edges)[0] / len(p_train)
    te = np.histogram(p_test, edges)[0] / len(p_test)
    bins = [f"{int(edges[i]*100)}-{int(edges[i+1]*100)}%" for i in range(len(edges)-1)]
    return dict(bin_data=[dict(bin=b, train_pct=round(float(t), 4), test_pct=round(float(s), 4))
                          for b, t, s in zip(bins, tr, te)],
                psi=round(psi(p_train, p_test), 4))


def politica_backtest(train, val, holdout, num, cat):
    cols = num + cat
    base = build_pipelines(num, cat)["LogReg"]; base.fit(train[cols], train[COL_TARGET])
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        base.predict_proba(val[cols])[:, 1], val[COL_TARGET])
    p = iso.predict(base.predict_proba(holdout[cols])[:, 1])
    G, L = economia(holdout); G, L = G.values, L.values
    y = holdout[COL_TARGET].values
    real = np.where(y == 0, G, -L)
    exposicion = holdout["monto_solicitado"].sum()

    def fila(nombre, aprobar):
        g = real[aprobar].sum()
        return dict(Politica=nombre, aprobacion=round(aprobar.mean()*100, 1),
                    mora=round(y[aprobar].mean()*100, 2) if aprobar.sum() else 0.0,
                    ganancia=int(g), roi=round(g/exposicion*100, 2))

    ev = (1-p)*G - p*L
    todo = np.ones(len(y), bool)
    filas = [fila("Aprobar todo (status quo)", todo),
             fila("Política por valor esperado", ev > 0)]
    # curva ganancia vs umbral global (para visualizar el óptimo)
    ths = np.linspace(0.02, 0.5, 25)
    curva = [dict(umbral=round(t, 3), ganancia=int(real[p < t].sum())) for t in ths]
    return filas, curva


# ---------------------------------------------------------------------------
def build_html(data):
    j = json.dumps(data)
    return r"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Andina Crédito — Panel MLOps: Entrenamiento, Drift y Política</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{--bg:#0b0f19;--bg2:#111827;--card:rgba(30,41,59,.7);--bd:rgba(255,255,255,.08);
--tx:#f8fafc;--tx2:#94a3b8;--blue:#38bdf8;--purple:#818cf8;--green:#34d399;--amber:#fbbf24;--rose:#fb7185;}
*{margin:0;padding:0;box-sizing:border-box;font-family:'Outfit',-apple-system,Segoe UI,Roboto,sans-serif;}
body{background:var(--bg);color:var(--tx);min-height:100vh;
background-image:radial-gradient(at 10% 10%,rgba(56,189,248,.08),transparent 50%),radial-gradient(at 90% 90%,rgba(129,140,248,.08),transparent 50%);}
header{background:rgba(17,24,39,.85);border-bottom:1px solid var(--bd);padding:1.2rem 2rem;}
header h1{font-size:1.3rem;color:var(--blue);}header p{color:var(--tx2);font-size:.85rem;margin-top:2px;}
.wrap{max-width:1100px;margin:0 auto;padding:1.5rem 2rem;}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:1.5rem;}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:1rem 1.2rem;}
.kpi .v{font-size:1.6rem;font-weight:700;}.kpi .l{color:var(--tx2);font-size:.78rem;margin-top:4px;}
.tabs{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem;}
.tab-btn{background:var(--bg2);color:var(--tx2);border:1px solid var(--bd);border-radius:8px;
padding:.5rem 1rem;cursor:pointer;font-size:.85rem;}
.tab-btn.active{color:var(--tx);border-color:var(--blue);background:rgba(56,189,248,.12);}
.tab-content{display:none;}.tab-content.active{display:block;}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:1.2rem;margin-bottom:1.2rem;}
.card h2{font-size:1rem;margin-bottom:.3rem;}.card .desc{color:var(--tx2);font-size:.82rem;margin-bottom:1rem;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;}@media(max-width:760px){.grid2{grid-template-columns:1fr;}}
table{width:100%;border-collapse:collapse;font-size:.82rem;}
th,td{padding:8px 10px;text-align:center;border-bottom:1px solid var(--bd);}
th{color:var(--tx2);font-weight:600;}td:first-child,th:first-child{text-align:left;}
code{background:rgba(255,255,255,.06);padding:1px 6px;border-radius:4px;font-size:.8rem;}
.badge{padding:2px 9px;border-radius:99px;font-size:.72rem;font-weight:600;}
.badge-green{background:rgba(52,211,153,.15);color:var(--green);}
.badge-amber{background:rgba(251,191,36,.15);color:var(--amber);}
.badge-red{background:rgba(251,113,133,.15);color:var(--rose);}
.champ{background:rgba(56,189,248,.10);font-weight:600;}
.grid2>*{min-width:0;}
.chart-box{position:relative;height:280px;min-width:0;}
.chart-box canvas{width:100%!important;height:100%!important;}
canvas{max-width:100%;}
.note{color:var(--tx2);font-size:.78rem;margin-top:.6rem;border-left:3px solid var(--blue);padding-left:.8rem;}
</style></head><body>
<header><h1>Andina Crédito — Panel MLOps</h1>
<p>Monitoreo de entrenamiento (challenger), detección de drift y política de aprobación · data real del proyecto</p></header>
<div class="wrap">
<div class="kpis" id="kpis"></div>
<div class="tabs">
<button class="tab-btn active" onclick="tab(event,'t-modelos')">Entrenamiento &amp; modelos</button>
<button class="tab-btn" onclick="tab(event,'t-drift')">Detección de drift</button>
<button class="tab-btn" onclick="tab(event,'t-politica')">Política de aprobación</button>
<button class="tab-btn" onclick="tab(event,'t-gob')">Gobernanza</button>
</div>

<div id="t-modelos" class="tab-content active">
  <div class="card"><h2>Curvas de entrenamiento — XGBoost (challenger)</h2>
  <div class="desc">El modelo productivo es la regresión logística (interpretable). El XGBoost es el
  challenger de boosting; estas son sus curvas reales por iteración, útiles para vigilar sobreajuste.</div>
  <div class="grid2"><div class="chart-box"><canvas id="cLogloss"></canvas></div><div class="chart-box"><canvas id="cAuc"></canvas></div></div>
  <div class="note">Si la curva de validación se separa de la de entrenamiento, aparece sobreajuste: ahí conviene early stopping.</div></div>
  <div class="card"><h2>Comparativo de modelos (holdout out-of-time)</h2>
  <div class="desc">Métricas honestas sobre datos no vistos, con el set sin la variable con leakage.</div>
  <table id="tModelos"></table>
  <div class="note" id="ensNote"></div></div>
</div>

<div id="t-drift" class="tab-content">
  <div class="card"><h2>Estado de drift de variables (PSI train → test)</h2>
  <div class="desc">PSI &lt; 0.10 estable · 0.10–0.25 advertencia · &ge; 0.25 crítico.</div>
  <div class="grid2"><div class="chart-box"><canvas id="cDonut"></canvas></div><div class="chart-box"><canvas id="cScore"></canvas></div></div></div>
  <div class="card"><h2>Detalle por variable</h2><table id="tDrift"></table></div>
</div>

<div id="t-politica" class="tab-content">
  <div class="card"><h2>Política de aprobación — backtest real</h2>
  <div class="desc">Sobre el holdout etiquetado: decisión con la probabilidad predicha, plata con el resultado real.</div>
  <table id="tPol"></table></div>
  <div class="card"><h2>Ganancia vs umbral global</h2>
  <div class="desc">La política recomendada usa un umbral por plazo (valor esperado); esta curva ilustra el óptimo con un umbral único.</div>
  <div class="chart-box"><canvas id="cGanancia"></canvas></div></div>
</div>

<div id="t-gob" class="tab-content">
  <div class="card"><h2>Registro y linaje</h2><table id="tGob"></table></div>
</div>
</div>

<script>
const D = """ + j + r""";
function tab(e,id){document.querySelectorAll('.tab-content').forEach(x=>x.classList.remove('active'));
document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active'));
document.getElementById(id).classList.add('active');e.currentTarget.classList.add('active');}
const AX={scales:{x:{grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#94a3b8'}},
y:{grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#94a3b8'}}},plugins:{legend:{labels:{color:'#94a3b8'}}},
responsive:true,maintainAspectRatio:false};

// KPIs
const champ=D.modelos.find(m=>m.Tipo==='Campeón');
const pol=D.politica.find(p=>p.Politica.includes('valor esperado'));
const todo=D.politica.find(p=>p.Politica.includes('todo'));
const crit=D.drift.filter(d=>d.psi>=0.25).length, adv=D.drift.filter(d=>d.psi>=0.10&&d.psi<0.25).length;
const kpis=[['AUC (holdout)',champ.AUC.toFixed(3)],['KS (holdout)',champ.KS.toFixed(3)],
['Ganancia política','$'+(pol.ganancia/1e6).toFixed(0)+'M'],
['Mejora vs aprobar todo','+$'+((pol.ganancia-todo.ganancia)/1e6).toFixed(0)+'M'],
['Variables con drift',(crit+adv)+' / '+D.drift.length]];
document.getElementById('kpis').innerHTML=kpis.map(k=>`<div class="kpi"><div class="v">${k[1]}</div><div class="l">${k[0]}</div></div>`).join('');

// Curvas XGBoost
const c=D.curvas;
new Chart(cLogloss,{type:'line',data:{labels:c.iter,datasets:[
{label:'Train logloss',data:c.train_logloss,borderColor:'#38bdf8',pointRadius:0,tension:.3},
{label:'Val logloss',data:c.val_logloss,borderColor:'#fb7185',pointRadius:0,tension:.3}]},
options:{...AX,plugins:{...AX.plugins,title:{display:true,text:'Log-loss por iteración',color:'#f8fafc'}}}});
new Chart(cAuc,{type:'line',data:{labels:c.iter,datasets:[
{label:'Train AUC',data:c.train_auc,borderColor:'#38bdf8',pointRadius:0,tension:.3},
{label:'Val AUC',data:c.val_auc,borderColor:'#34d399',pointRadius:0,tension:.3}]},
options:{...AX,plugins:{...AX.plugins,title:{display:true,text:'AUC por iteración',color:'#f8fafc'}}}});

// Tabla modelos
document.getElementById('tModelos').innerHTML='<tr><th>Modelo</th><th>AUC</th><th>KS</th><th>PR-AUC</th><th>Gini</th><th>Brier</th><th>ECE</th><th>Tipo</th></tr>'+
D.modelos.map(m=>`<tr class="${m.Tipo==='Campeón'?'champ':''}"><td>${m.Modelo}${m.Tipo==='Campeón'?' 🏆':''}</td>
<td>${m.AUC.toFixed(3)}</td><td>${m.KS.toFixed(3)}</td><td>${m.PR_AUC.toFixed(3)}</td><td>${m.Gini.toFixed(3)}</td>
<td>${m.Brier.toFixed(4)}</td><td>${(m.ECE*100).toFixed(2)}%</td><td><code>${m.Tipo}</code></td></tr>`).join('');
document.getElementById('ensNote').textContent='Correlación media entre modelos: '+D.corr+' (alta → el ensemble no aporta; se descarta).';

// Drift donut + tabla
new Chart(cDonut,{type:'doughnut',data:{labels:['Estable','Advertencia','Crítico'],
datasets:[{data:[D.drift.filter(d=>d.psi<0.10).length,adv,crit],backgroundColor:['#34d399','#fbbf24','#fb7185'],borderWidth:0}]},
options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#94a3b8'}},title:{display:true,text:'Estado de las variables',color:'#f8fafc'}}}});
new Chart(cScore,{type:'bar',data:{labels:D.score_drift.bin_data.map(b=>b.bin),datasets:[
{label:'Train',data:D.score_drift.bin_data.map(b=>(b.train_pct*100)),backgroundColor:'rgba(56,189,248,.6)'},
{label:'Test',data:D.score_drift.bin_data.map(b=>(b.test_pct*100)),backgroundColor:'rgba(129,140,248,.6)'}]},
options:{...AX,plugins:{...AX.plugins,title:{display:true,text:'Drift de score (PSI='+D.score_drift.psi+')',color:'#f8fafc'}}}});
const bcls=v=>v<0.10?'badge-green':(v<0.25?'badge-amber':'badge-red');
document.getElementById('tDrift').innerHTML='<tr><th>Variable</th><th>Tipo</th><th>PSI</th><th>Estado</th><th>KS</th><th>μ train</th><th>μ test</th><th>Nulos tr/te</th></tr>'+
D.drift.map(r=>`<tr><td><code>${r.feature}</code></td><td>${r.tipo}</td><td><b>${r.psi}</b></td>
<td><span class="badge ${bcls(r.psi)}">${r.status}</span></td>
<td>${isNaN(r.ks_statistic)?'-':r.ks_statistic}</td>
<td>${isNaN(r.train_mean)?'-':r.train_mean.toLocaleString()}</td>
<td>${isNaN(r.test_mean)?'-':r.test_mean.toLocaleString()}</td>
<td>${r.train_nulls_pct}% / ${r.test_nulls_pct}%</td></tr>`).join('');

// Política
document.getElementById('tPol').innerHTML='<tr><th>Política</th><th>Aprobación</th><th>Mora cartera</th><th>Ganancia (CLP)</th><th>ROI</th><th></th></tr>'+
D.politica.map(p=>{const rec=p.Politica.includes('valor esperado');
return `<tr class="${rec?'champ':''}"><td><b>${p.Politica}</b></td><td>${p.aprobacion}%</td><td>${p.mora}%</td>
<td><b>$${(p.ganancia/1e6).toFixed(1)}M</b></td><td>${p.roi}%</td>
<td><span class="badge ${rec?'badge-green':'badge-amber'}">${rec?'Recomendada':'Status quo'}</span></td></tr>`;}).join('');
new Chart(cGanancia,{type:'line',data:{labels:D.curva.map(c=>(c.umbral*100).toFixed(0)+'%'),
datasets:[{label:'Ganancia (MM CLP)',data:D.curva.map(c=>c.ganancia/1e6),borderColor:'#34d399',backgroundColor:'rgba(52,211,153,.1)',fill:true,tension:.3,pointRadius:0}]},
options:{...AX,plugins:{...AX.plugins,title:{display:true,text:'Ganancia según umbral global de aprobación',color:'#f8fafc'}}}});

// Gobernanza
document.getElementById('tGob').innerHTML=D.gob.map(r=>`<tr><td>${r[0]}</td><td><code>${r[1]}</code></td></tr>`).join('');
</script></body></html>"""


def main():
    train_g = pd.read_parquet(GOLD_DIR / "train_gold.parquet")
    test_g = pd.read_parquet(GOLD_DIR / "test_gold.parquet")
    num, cat = columnas_num_cat(train_g)
    train, val, holdout = split_temporal_3(train_g)

    modelos, corr = comparativo_modelos(train, val, holdout, num, cat)
    filas_pol, curva = politica_backtest(train, val, holdout, num, cat)

    # drift sobre TODAS las features (incluye la excluida, a modo de monitoreo)
    excl = {COL_ID, COL_TARGET, COL_FECHA}
    num_all = [c for c in train_g.columns if c not in excl and pd.api.types.is_numeric_dtype(train_g[c])]
    cat_all = [c for c in train_g.columns if c not in excl and c not in num_all]

    data = dict(
        curvas=curvas_xgboost(train, val, num, cat),
        modelos=modelos, corr=corr,
        drift=tabla_drift(train_g, test_g, num_all, cat_all),
        score_drift=score_drift(train, val, holdout, test_g, num, cat),
        politica=filas_pol, curva=curva,
        gob=[["Modelo productivo", "Regresión logística calibrada (isotónica)"],
             ["Challenger", "XGBoost (boosting)"],
             ["Dataset entrenamiento", "data/raw/train.csv (45.300 registros)"],
             ["Variable excluida (leakage)", ", ".join(COLS_LEAKAGE_SOSPECHADO)],
             ["Validación", "Out-of-time de 3 vías (train/val/holdout)"]],
    )
    salida = PROJECT_ROOT / "reports" / "mlops_dashboard.html"
    salida.write_text(build_html(data), encoding="utf-8")
    print(f"Dashboard generado -> {salida}  ({salida.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()