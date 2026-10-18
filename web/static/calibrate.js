// Floor calibration: click floor marks on the camera picture, type their metres.
// Picture points are kept as 0..1 fractions, so the picture can be any size.
(function () {
  const C = window.CALIB;
  const $ = (id) => document.getElementById(id);
  const img = $("snap"), canvas = $("overlay"), ctx = canvas.getContext("2d");
  const MAX_POINTS = 12, HIT_PX = 14;

  let points = C.initial.image.map((p, i) => ({
    x: p[0], y: p[1],
    X: String(C.initial.floor[i][0]), Y: String(C.initial.floor[i][1]),
    err: C.initial.errors[i],
  }));
  let grid = C.initial.grid || [];
  let dragging = -1;

  function message(text, bad) {
    const m = $("msg");
    m.textContent = text; m.hidden = !text;
    m.style.borderLeftColor = bad ? "var(--bad)" : "var(--ok)";
  }

  // ---------------------------------------------------------------- picture
  function loadPicture() {
    img.src = img.dataset.src + "?t=" + Date.now();
  }
  img.addEventListener("load", draw);
  img.addEventListener("error", () => message("ยังไม่มีภาพจากกล้อง (กล้องหยุดอยู่ หรือเพิ่งเริ่ม รอสักครู่แล้วกด \"ภาพใหม่\")", true));
  window.addEventListener("resize", draw);
  $("refresh").addEventListener("click", loadPicture);

  function size() {
    const w = img.clientWidth, h = img.clientHeight, dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  function draw() {
    const { w, h } = size();
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(40, 220, 120, 0.85)";
    grid.forEach((line) => {
      ctx.beginPath();
      line.forEach(([x, y], i) => (i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h)));
      ctx.stroke();
    });
    ctx.font = "bold 13px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    points.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(p.x * w, p.y * h, 11, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(255, 140, 0, 0.9)"; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();
      ctx.fillStyle = "#000"; ctx.fillText(String(i + 1), p.x * w, p.y * h + 0.5);
    });
  }

  // ---------------------------------------------------------------- clicking
  function at(ev) {
    const r = canvas.getBoundingClientRect();
    return { x: (ev.clientX - r.left) / r.width, y: (ev.clientY - r.top) / r.height, w: r.width, h: r.height };
  }
  canvas.addEventListener("pointerdown", (ev) => {
    if (!img.naturalWidth) return;
    const p = at(ev);
    dragging = points.findIndex((q) => Math.hypot((q.x - p.x) * p.w, (q.y - p.y) * p.h) < HIT_PX);
    if (dragging < 0) {
      if (points.length >= MAX_POINTS) { message(`ใส่ได้สูงสุด ${MAX_POINTS} จุด`, true); return; }
      points.push({ x: p.x, y: p.y, X: "", Y: "", err: null });
      dragging = points.length - 1;
      renderRows(true);
    }
    canvas.setPointerCapture(ev.pointerId);
    changed();
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (dragging < 0) return;
    const p = at(ev);
    points[dragging].x = Math.min(1, Math.max(0, p.x));
    points[dragging].y = Math.min(1, Math.max(0, p.y));
    draw();
  });
  canvas.addEventListener("pointerup", () => { dragging = -1; });

  function changed() { grid = []; points.forEach((p) => (p.err = null)); draw(); }

  // ---------------------------------------------------------------- table
  function renderRows(focusLast) {
    const body = $("rows");
    body.innerHTML = "";
    if (!points.length) {
      body.innerHTML = '<tr><td colspan="5" class="muted small">ยังไม่มีจุด คลิกบนภาพ</td></tr>';
    }
    points.forEach((p, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><b>${i + 1}</b></td>
        <td><input type="text" inputmode="decimal" value="${p.X}" data-i="${i}" data-k="X" style="width:62px"></td>
        <td><input type="text" inputmode="decimal" value="${p.Y}" data-i="${i}" data-k="Y" style="width:62px"></td>
        <td class="small">${p.err == null ? "" : p.err.toFixed(2)}</td>
        <td><button class="btn" data-del="${i}" title="ลบจุด">✕</button></td>`;
      body.appendChild(tr);
    });
    body.querySelectorAll("input").forEach((input) =>
      input.addEventListener("input", () => { points[input.dataset.i][input.dataset.k] = input.value; grid = []; draw(); }));
    body.querySelectorAll("[data-del]").forEach((b) =>
      b.addEventListener("click", () => { points.splice(Number(b.dataset.del), 1); renderRows(); changed(); }));
    if (focusLast && points.length) body.querySelector(`input[data-i="${points.length - 1}"][data-k="X"]`).focus();
    $("summary").textContent = points.length < 4 ? `ต้องมีอย่างน้อย 4 จุด (ตอนนี้ ${points.length})` : "";
  }

  // ---------------------------------------------------------------- saving
  async function post(body) {
    const res = await fetch(C.saveUrl, {
      method: "POST", body: JSON.stringify(body),
      headers: { "Content-Type": "application/json", "X-CSRFToken": C.csrf },
    });
    if (res.redirected) { location.reload(); return null; }
    return { ok: res.ok, data: await res.json().catch(() => ({ error: `HTTP ${res.status}` })) };
  }

  $("save").addEventListener("click", async () => {
    const floor = [];
    for (let i = 0; i < points.length; i++) {
      const X = parseFloat(String(points[i].X).replace(",", ".")), Y = parseFloat(String(points[i].Y).replace(",", "."));
      if (!Number.isFinite(X) || !Number.isFinite(Y)) { message(`จุดที่ ${i + 1}: ใส่ X และ Y เป็นตัวเลข (เมตร)`, true); return; }
      floor.push([X, Y]);
    }
    const reply = await post({ image: points.map((p) => [p.x, p.y]), floor });
    if (!reply) return;
    if (!reply.ok) { message(reply.data.error || "บันทึกไม่สำเร็จ", true); return; }
    grid = reply.data.grid;
    reply.data.errors.forEach((e, i) => (points[i].err = e));
    renderRows(); draw();
    message(`บันทึกแล้ว · คลาดเฉลี่ย ${reply.data.error_m.toFixed(2)} ม. ตรวจว่าเส้นตารางสีเขียวตรงกับพื้นจริง`, false);
  });

  $("clear").addEventListener("click", async () => {
    if (!confirm("ลบการปรับเทียบของกล้องนี้?")) return;
    const reply = await post({ clear: true });
    if (reply && reply.ok) { points = []; grid = []; renderRows(); draw(); message("ลบแล้ว กล้องนี้จะไม่ถูกรวมกับกล้องอื่น", false); }
  });

  renderRows();
  loadPicture();
  if (C.initial.error_m != null) message(`ปรับเทียบไว้แล้ว · คลาดเฉลี่ย ${C.initial.error_m.toFixed(2)} ม.`, false);
})();
