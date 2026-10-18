// Top-down floor map: everyone the joined cameras see, in metres.
(function () {
  const canvas = document.getElementById("map"), ctx = canvas.getContext("2d");
  const CAMERA_COLORS = ["#2459d6", "#d6246e", "#1e8a4c", "#b26b00", "#7a3fd1", "#0a8f9a"];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const personColor = (gid) => `hsl(${(gid * 137.5) % 360}, 70%, 45%)`;
  let latest = { people: [], cameras: [] };
  let bounds = null; // grows only, so the map does not jump around

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function updateBounds(data) {
    const xs = [], ys = [];
    data.cameras.forEach((c) => c.floor.forEach(([x, y]) => { xs.push(x); ys.push(y); }));
    data.people.forEach((p) => { if (p.x != null) { xs.push(p.x); ys.push(p.y); } });
    if (!xs.length) return;
    const b = { x0: Math.min(...xs) - 1, x1: Math.max(...xs) + 1, y0: Math.min(...ys) - 1, y1: Math.max(...ys) + 1 };
    bounds = bounds ? { x0: Math.min(bounds.x0, b.x0), x1: Math.max(bounds.x1, b.x1),
                        y0: Math.min(bounds.y0, b.y0), y1: Math.max(bounds.y1, b.y1) } : b;
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = Math.max(320, Math.min(window.innerHeight * 0.7, w * 0.75));
    canvas.style.height = h + "px";
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!bounds) {
      ctx.fillStyle = css("--muted"); ctx.font = "14px system-ui, sans-serif";
      ctx.fillText("ยังไม่มีข้อมูล", 16, 28);
      return;
    }
    const pad = 28;
    const scale = Math.min((w - 2 * pad) / (bounds.x1 - bounds.x0), (h - 2 * pad) / (bounds.y1 - bounds.y0));
    // x to the right, y upward (like a map)
    const X = (x) => pad + (x - bounds.x0) * scale;
    const Y = (y) => h - pad - (y - bounds.y0) * scale;

    // 1-metre grid
    const step = (bounds.x1 - bounds.x0) > 30 ? 5 : 1;
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
    ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif";
    for (let x = Math.ceil(bounds.x0 / step) * step; x <= bounds.x1; x += step) {
      ctx.beginPath(); ctx.moveTo(X(x), Y(bounds.y0)); ctx.lineTo(X(x), Y(bounds.y1)); ctx.stroke();
      ctx.fillText(`${x}`, X(x) + 2, h - 8);
    }
    for (let y = Math.ceil(bounds.y0 / step) * step; y <= bounds.y1; y += step) {
      ctx.beginPath(); ctx.moveTo(X(bounds.x0), Y(y)); ctx.lineTo(X(bounds.x1), Y(y)); ctx.stroke();
      ctx.fillText(`${y}`, 4, Y(y) - 2);
    }
    ctx.fillText("เมตร", w - 34, h - 8);

    // calibration marks of each camera
    latest.cameras.forEach((c, i) => {
      ctx.fillStyle = CAMERA_COLORS[i % CAMERA_COLORS.length];
      const half = Math.max(3, 7 - 2 * i); // later cameras smaller, so shared marks stay visible
      c.floor.forEach(([x, y]) => ctx.fillRect(X(x) - half, Y(y) - half, 2 * half, 2 * half));
    });

    // people
    latest.people.forEach((p) => {
      if (p.x == null) return;
      const r = Math.max(7, 0.25 * scale);
      ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), r, 0, 2 * Math.PI);
      ctx.fillStyle = personColor(p.gid); ctx.fill();
      ctx.lineWidth = p.cameras.length > 1 ? 3 : 1; ctx.strokeStyle = css("--text"); ctx.stroke();
      ctx.fillStyle = css("--text"); ctx.font = "bold 13px system-ui, sans-serif";
      ctx.fillText(`G${p.gid}${p.name ? " " + p.name : ""}`, X(p.x) + r + 4, Y(p.y) + 4);
    });
  }

  function renderList(data) {
    document.getElementById("count").textContent = `(${data.people.length})`;
    const list = document.getElementById("people");
    list.innerHTML = data.people.length ? data.people.map((p) => `
      <li><span class="chip" style="background:${personColor(p.gid)}"></span>
        <b>G${p.gid} ${esc(p.name || "ไม่ทราบชื่อ")}</b>
        ${p.name && p.name_camera ? `<span class="small muted">(จำหน้าได้จาก ${esc(p.name_camera)})</span>` : ""}<br>
        <span class="small">เห็นโดย: ${p.cameras.map(esc).join(", ")}${p.cameras.length > 1 ? " ✓ รวมแล้ว" : ""}</span>
        ${p.x != null ? `<span class="small muted"> · (${p.x}, ${p.y}) ม.</span>` : ""}</li>`).join("")
      : '<li class="muted">ไม่มีคน</li>';
    document.getElementById("legend").innerHTML = data.cameras.map((c, i) =>
      `<span class="chip" style="background:${CAMERA_COLORS[i % CAMERA_COLORS.length]}"></span> จุดปรับเทียบของ ${esc(c.name)}${c.running ? "" : " (หยุดอยู่)"}`).join(" &nbsp; ");
  }

  async function poll() {
    try {
      const res = await fetch(window.FLOOR_URL, { cache: "no-store" });
      if (res.redirected || !res.ok) return;
      latest = await res.json();
      updateBounds(latest);
      renderList(latest);
      draw();
    } catch (e) { /* server restarting */ }
  }
  window.addEventListener("resize", draw);
  poll();
  setInterval(poll, 500);
})();
