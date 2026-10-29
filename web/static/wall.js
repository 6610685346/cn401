/* "ดูหลายกล้อง": every tile fetches its camera's next picture (frame.jpg?after=id)
   in its own loop, at most N per second. Short requests instead of one MJPEG
   stream per camera, so many cameras do not use up the browser's 6 connections. */
(function () {
  const wall = document.getElementById("wall");
  if (!wall) return;
  const tiles = [...wall.querySelectorAll(".tile")];
  const store = {
    get(k, d) { try { const v = localStorage.getItem("wall." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("wall." + k, JSON.stringify(v)); } catch (e) { /* private window */ } },
  };
  let fps = Number(store.get("fps", 8));
  const fpsSelect = document.getElementById("wall-fps");
  fpsSelect.value = String(fps);
  fpsSelect.addEventListener("change", () => { fps = Number(fpsSelect.value); store.set("fps", fps); });

  // ---- layout: columns, hidden cameras, focus on one tile, full screen
  const segButtons = [...document.querySelectorAll(".wall-controls [data-cols]")];
  const setCols = (cols) => {
    wall.dataset.cols = cols;
    segButtons.forEach((b) => b.classList.toggle("on", b.dataset.cols === cols));
    store.set("cols", cols);
  };
  segButtons.forEach((b) => b.addEventListener("click", () => setCols(b.dataset.cols)));
  setCols(store.get("cols", "auto"));

  const hidden = new Set(store.get("hidden", []).map(String));
  const applyHidden = () => {
    tiles.forEach((t) => { t.hidden = hidden.has(t.dataset.camera); });
    document.querySelectorAll("[data-pick]").forEach((b) => b.classList.toggle("on", !hidden.has(b.dataset.pick)));
    const shown = tiles.filter((t) => !t.hidden).length;
    wall.style.setProperty("--n", Math.max(1, shown));
  };
  document.querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", () => {
    hidden.has(b.dataset.pick) ? hidden.delete(b.dataset.pick) : hidden.add(b.dataset.pick);
    store.set("hidden", [...hidden]);
    applyHidden();
  }));
  applyHidden();

  const toggleFocus = (tile) => {
    const on = !tile.classList.contains("focus");
    tiles.forEach((t) => t.classList.remove("focus"));
    wall.classList.toggle("has-focus", on);
    if (on) tile.classList.add("focus");
  };
  tiles.forEach((t) => {
    t.addEventListener("click", (e) => { if (!e.target.closest("a, button, form")) toggleFocus(t); });
    t.addEventListener("keydown", (e) => { if (e.key === "Enter") toggleFocus(t); });
  });
  const full = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else if (wall.requestFullscreen) wall.requestFullscreen().catch(() => {});
  };
  document.getElementById("wall-full").addEventListener("click", full);
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, select, textarea")) return;
    if (e.key === "f" || e.key === "F") full();
    if (e.key === "Escape") { tiles.forEach((t) => t.classList.remove("focus")); wall.classList.remove("has-focus"); }
  });

  // ---- pictures
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  async function loop(tile) {
    const img = tile.querySelector("img");
    const empty = tile.querySelector(".tile-empty");
    const label = tile.querySelector('[data-field="empty"]');
    let last = -1, url = null;
    for (;;) {
      if (tile.hidden || document.hidden || tile.dataset.running !== "1") { await sleep(800); continue; }
      const started = performance.now();
      try {
        const r = await fetch(`${tile.dataset.frameUrl}?after=${last}`, { cache: "no-store" });
        if (r.status === 200) {
          last = Number(r.headers.get("X-Frame-Id") || last);
          const next = URL.createObjectURL(await r.blob());
          img.src = next;
          await img.decode().catch(() => {});
          if (url) URL.revokeObjectURL(url);
          url = next;
          img.hidden = false; empty.hidden = true;
        } else if (r.status !== 204) {
          throw new Error(String(r.status));
        }
      } catch (e) {
        img.hidden = true; empty.hidden = false;
        if (label) label.textContent = "ยังไม่มีภาพ กำลังลองใหม่…";
        await sleep(2000);
        continue;
      }
      await sleep(Math.max(0, 1000 / fps - (performance.now() - started)));
    }
  }
  tiles.forEach(loop);

  // ---- status (people, fps) every 2 s
  async function status() {
    try {
      const data = await (await fetch(wall.dataset.statusUrl, { cache: "no-store" })).json();
      tiles.forEach((tile) => {
        const cam = data.cameras[tile.dataset.camera];
        if (!cam) return;
        tile.dataset.running = cam.status === "running" ? "1" : "0";
        const pill = tile.querySelector('[data-field="status"]');
        pill.className = `pill s-${cam.status}`;
        pill.textContent = cam.status;
        tile.querySelector('[data-field="people_now"]').textContent = cam.people_now ?? "-";
        tile.querySelector('[data-field="fps"]').textContent = cam.status === "running" ? cam.fps : "-";
        if (cam.status !== "running") {
          tile.querySelector("img").hidden = true;
          tile.querySelector(".tile-empty").hidden = false;
          const label = tile.querySelector('[data-field="empty"]');
          if (label) label.textContent = cam.status === "stopped" ? "กล้องนี้หยุดอยู่" : (cam.message || "กำลังเชื่อมต่อ…");
        }
      });
    } catch (e) { /* server busy: next time */ }
    setTimeout(status, 2000);
  }
  status();
})();
