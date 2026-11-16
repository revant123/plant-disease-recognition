// Front end for the Plant Disease Recognition API.
// The API base URL comes from config.js (written at build time from the API_URL env var),
// and can be overridden for testing with ?api=https://... in the page URL.
const API = (new URLSearchParams(location.search).get("api") || window.API_URL || "").replace(/\/+$/, "");

const SAMPLES = [
  ["01_potato_late_blight.jpg", "Potato — late blight (lab)"],
  ["02_tomato_healthy.jpg", "Tomato — healthy (lab)"],
  ["05_grape_black_rot.jpg", "Grape — black rot (lab)"],
  ["01_apple_apple_scab.jpg", "Apple — scab (field)"],
  ["05_corn_common_rust.jpg", "Corn — common rust (field)"],
  ["06_squash_powdery_mildew.jpg", "Squash — powdery mildew (field)"],
];

const $ = (id) => document.getElementById(id);
const statusEl = $("status");

function setStatus(kind, text) {
  statusEl.className = `status status-${kind}`;
  statusEl.textContent = text;
}

// Poll /health so the user can see when the backend is awake (Railway may take a moment after a deploy).
async function checkHealth(tries = 0) {
  if (!API) return setStatus("bad", "API URL not set");
  try {
    const r = await fetch(`${API}/health`, { cache: "no-store" });
    const d = await r.json();
    if (d.status === "ok") return setStatus("ok", "Model ready");
    setStatus("wait", "Model loading…");
  } catch {
    setStatus(tries > 3 ? "bad" : "wait", tries > 3 ? "API unreachable" : "Waking API…");
  }
  if (tries < 30) setTimeout(() => checkHealth(tries + 1), 3000);
}

function pct(x) { return `${(x * 100).toFixed(1)}%`; }

async function predict(blob, name = "leaf.jpg") {
  const result = $("result");
  result.hidden = false;
  $("orig").src = URL.createObjectURL(blob);
  $("cam").removeAttribute("src");
  $("spinner").hidden = false;
  $("verdict").textContent = "";
  $("verdict").className = "verdict";
  $("top3").innerHTML = "";
  $("error").hidden = true;
  result.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    if (!API) throw new Error("The API URL is not configured. Set API_URL in Vercel and redeploy.");
    const form = new FormData();
    form.append("file", blob, name);
    const r = await fetch(`${API}/predict`, { method: "POST", body: form });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `Server error (${r.status})`);

    $("cam").src = d.gradcam_png;
    const p = d.prediction;
    const v = $("verdict");
    v.className = `verdict ${p.healthy ? "healthy" : "sick"}`;
    v.innerHTML = `<strong>${p.crop}</strong> — ${p.condition} <span class="conf">· ${pct(p.confidence)} confidence</span>`;

    $("top3").innerHTML = "<div class='muted small'>Top-3 predictions</div>" + d.top3.map((t) => `
      <div class="bar-row">
        <div class="bar-label"><span>${t.label}</span><span>${pct(t.confidence)}</span></div>
        <div class="bar"><span style="width:${(t.confidence * 100).toFixed(1)}%"></span></div>
      </div>`).join("");
    setStatus("ok", "Model ready");
  } catch (e) {
    const msg = e instanceof TypeError ? "Could not reach the API. It may be waking up — try again in a few seconds." : e.message;
    $("error").textContent = msg;
    $("error").hidden = false;
  } finally {
    $("spinner").hidden = true;
  }
}

// Upload + drag-and-drop
const drop = $("drop");
$("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) predict(f, f.name);
  e.target.value = "";
});
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) predict(f, f.name);
});

// Sample images
const row = $("samples");
SAMPLES.forEach(([file, label]) => {
  const b = document.createElement("button");
  b.type = "button";
  b.title = label;
  b.setAttribute("aria-label", `Try sample: ${label}`);
  b.innerHTML = `<img src="samples/${file}" alt="">`;
  b.onclick = async () => {
    const blob = await (await fetch(`samples/${file}`)).blob();
    predict(blob, file);
  };
  row.appendChild(b);
});

checkHealth();
