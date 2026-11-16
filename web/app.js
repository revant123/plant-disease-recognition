// Front end for the Plant Disease Recognition API.
// The API base URL comes from config.js (written at build time from the API_URL env var),
// and can be overridden for testing with ?api=https://... in the page URL.
const API = (new URLSearchParams(location.search).get("api") || window.API_URL || "").replace(/\/+$/, "");

const SAMPLES = [
  ["01_potato_late_blight.jpg", "Potato, lab photo"],
  ["02_tomato_healthy.jpg", "Tomato, lab photo"],
  ["05_grape_black_rot.jpg", "Grape, lab photo"],
  ["01_apple_apple_scab.jpg", "Apple, field photo"],
  ["05_corn_common_rust.jpg", "Corn, field photo"],
  ["06_squash_powdery_mildew.jpg", "Squash, field photo"],
];

const $ = (id) => document.getElementById(id);
const statusEl = $("status");

function setStatus(kind, text) {
  statusEl.className = `status status-${kind}`;
  statusEl.textContent = text;
}

// Poll /health so people can see when the model is awake (Railway can take a moment after a deploy).
async function checkHealth(tries = 0) {
  if (!API) return setStatus("bad", "Model address not set");
  try {
    const r = await fetch(`${API}/health`, { cache: "no-store" });
    const d = await r.json();
    if (d.status === "ok") return setStatus("ok", "Model ready");
    setStatus("wait", "Model loading");
  } catch {
    setStatus(tries > 3 ? "bad" : "wait", tries > 3 ? "Model offline" : "Waking the model");
  }
  if (tries < 30) setTimeout(() => checkHealth(tries + 1), 3000);
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function predict(blob, name = "leaf.jpg") {
  const result = $("result");
  result.hidden = false;
  $("orig").src = URL.createObjectURL(blob);
  $("cam").removeAttribute("src");
  $("spinner").hidden = false;
  $("label").hidden = true;
  $("error").hidden = true;
  result.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    if (!API) throw new Error("The site doesn't know where the model is. Set API_URL in Vercel and redeploy.");
    const form = new FormData();
    form.append("file", blob, name);
    const r = await fetch(`${API}/predict`, { method: "POST", body: form });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `The model returned an error (${r.status}). Try another photo.`);

    $("cam").src = d.gradcam_png;
    const p = d.prediction;
    $("crop").textContent = p.crop;
    $("condition").textContent = p.healthy ? "Healthy" : p.condition;
    $("confidence").textContent = pct(p.confidence);
    const stamp = $("stamp");
    stamp.className = `stamp ${p.healthy ? "healthy" : "sick"}`;
    stamp.textContent = p.healthy ? "Healthy" : "Disease found";

    $("top3").innerHTML = "<p class='top3-title'>Top three possibilities</p>" + d.top3.map((t) => `
      <div class="bar-row">
        <div class="bar-label"><span>${esc(t.label)}</span><span>${pct(t.confidence)}</span></div>
        <div class="bar"><span style="width:${(t.confidence * 100).toFixed(1)}%"></span></div>
      </div>`).join("");
    $("label").hidden = false;
    setStatus("ok", "Model ready");
  } catch (e) {
    $("error").textContent = e instanceof TypeError
      ? "Couldn't reach the model. It may be waking up, so try again in a few seconds."
      : e.message;
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

// Sample leaves
const row = $("samples");
SAMPLES.forEach(([file, label]) => {
  const b = document.createElement("button");
  b.type = "button";
  b.setAttribute("aria-label", `Try the sample: ${label}`);
  b.innerHTML = `<img src="samples/${file}" alt=""><span>${label}</span>`;
  b.onclick = async () => {
    const blob = await (await fetch(`samples/${file}`)).blob();
    predict(blob, file);
  };
  row.appendChild(b);
});

checkHealth();
