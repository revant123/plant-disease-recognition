// Vercel build step: copy the static site into dist/ and write config.js from the API_URL env var.
// Set API_URL in Vercel → Project → Settings → Environment Variables to your Railway URL,
// e.g. https://plant-disease-api-production.up.railway.app
import { cpSync, mkdirSync, rmSync, writeFileSync } from "node:fs";

const out = "dist";
rmSync(out, { recursive: true, force: true });
mkdirSync(out);
for (const f of ["index.html", "style.css", "app.js", "samples"]) cpSync(f, `${out}/${f}`, { recursive: true });

const api = (process.env.API_URL || "").trim().replace(/\/+$/, "");
writeFileSync(`${out}/config.js`, `window.API_URL = ${JSON.stringify(api)};\n`);
console.log(api ? `API_URL = ${api}` : "WARNING: API_URL is not set — the site will not be able to reach the model.");
