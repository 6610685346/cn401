// Door line: two points on the camera picture (0..1) + which side is inside.
(function () {
  const D = window.DOOR;
  const $ = (id) => document.getElementById(id);
  const img = $("snap"), canvas = $("overlay"), ctx = canvas.getContext("2d");
  const HIT_PX = 16;
  let pts = D.initial ? [{ x: D.initial.a[0], y: D.initial.a[1] }, { x: D.initial.b[0], y: D.initial.b[1] }] : [];
  let inside = (D.initial && D.initial.inside) || "left";
  let dragging = -1;

  function message(text, bad) {
    const m = $("msg");
    m.textContent = text; m.hidden = !text;
    m.style.borderLeftColor = bad ? "var(--bad)" : "var(--ok)";
    m.style.background = bad ? "var(--bad-soft)" : "var(--ok-soft)";
  }
  const load = () => { img.src = img.dataset.src + "?t=" + Date.now(); };
  img.addEventListener("load", draw);
  img.addEventListener("error", () => { message("ยังไม่มีภาพจากกล้อง (กล้องหยุดอยู่ หรือกำลังรีสตาร์ท) รอสักครู่แล้วกด \"ภาพใหม่\"", true); });
  window.addEventListener("resize", draw);
  $("refresh").addEventListener("click", load);

  function size() {
    const w = img.clientWidth, h = img.clientHeight, dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  function label(text, x, y, color) {
    ctx.font = "bold 14px system-ui, sans-serif";
    const tw = ctx.measureText(text).width + 14;
    ctx.fillStyle = color; ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(x - tw / 2, y - 12, tw, 24, 12) : ctx.rect(x - tw / 2, y - 12, tw, 24);
    ctx.fill();
    ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, x, y + 1);
  }

  function draw() {
    const { w, h } = size();
    ctx.clearRect(0, 0, w, h);
    if (pts.length === 2) {
      const [a, b] = pts.map((p) => ({ x: p.x * w, y: p.y * h }));
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(0, 0, 0, .45)"; ctx.lineWidth = 9;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      ctx.strokeStyle = "#ffb020"; ctx.lineWidth = 5;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      // normal pointing to the "left" of a->b in picture coordinates
      const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
      let nx = (b.y - a.y) / len, ny = -(b.x - a.x) / len;
      if (inside === "right") { nx = -nx; ny = -ny; }
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, L = Math.min(70, Math.max(40, len * 0.3));
      const tx = mx + nx * L, ty = my + ny * L;
      ctx.strokeStyle = "#22c55e"; ctx.fillStyle = "#22c55e"; ctx.lineWidth = 5;
      ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(tx, ty); ctx.stroke();
      const ang = Math.atan2(ty - my, tx - mx);
      ctx.beginPath(); ctx.moveTo(tx + Math.cos(ang) * 10, ty + Math.sin(ang) * 10);
      ctx.lineTo(tx + Math.cos(ang + 2.4) * 14, ty + Math.sin(ang + 2.4) * 14);
      ctx.lineTo(tx + Math.cos(ang - 2.4) * 14, ty + Math.sin(ang - 2.4) * 14); ctx.closePath(); ctx.fill();
      label("ข้างใน · เข้า", mx + nx * (L + 30), my + ny * (L + 30), "rgba(22, 163, 74, .92)");
      label("ข้างนอก", mx - nx * 44, my - ny * 44, "rgba(100, 116, 139, .92)");
    }
    pts.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(p.x * w, p.y * h, 10, 0, 2 * Math.PI);
      ctx.fillStyle = "#ff8c00"; ctx.fill(); ctx.lineWidth = 2.5; ctx.strokeStyle = "#fff"; ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.font = "bold 11px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(i ? "B" : "A", p.x * w, p.y * h + 0.5);
    });
  }

  function at(ev) {
    const r = canvas.getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height)), w: r.width, h: r.height };
  }
  canvas.addEventListener("pointerdown", (ev) => {
    if (!img.naturalWidth) return;
    const p = at(ev);
    dragging = pts.findIndex((q) => Math.hypot((q.x - p.x) * p.w, (q.y - p.y) * p.h) < HIT_PX);
    if (dragging < 0) {
      if (pts.length >= 2) return; // drag the ends, or press "วาดใหม่"
      pts.push({ x: p.x, y: p.y });
      dragging = pts.length - 1;
    }
    canvas.setPointerCapture(ev.pointerId);
    message(pts.length < 2 ? "คลิกอีกจุดเพื่อจบเส้น" : "");
    draw();
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (dragging < 0) return;
    const p = at(ev); pts[dragging].x = p.x; pts[dragging].y = p.y; draw();
  });
  canvas.addEventListener("pointerup", () => { dragging = -1; });

  $("flip").addEventListener("click", () => { inside = inside === "left" ? "right" : "left"; draw(); });
  $("redraw").addEventListener("click", () => { pts = []; message("คลิกจุดแรกของเส้นประตูบนภาพ"); draw(); });

  async function send(body) {
    const res = await fetch(D.saveUrl, { method: "POST", headers: { "Content-Type": "application/json", "X-CSRFToken": D.csrf }, body: JSON.stringify(body) });
    return [res.ok, await res.json().catch(() => ({}))];
  }
  $("save").addEventListener("click", async () => {
    if (pts.length !== 2) { message("ต้องวาดเส้น 2 จุดก่อน", true); return; }
    const [ok, data] = await send({ a: [pts[0].x, pts[0].y], b: [pts[1].x, pts[1].y], inside });
    if (!ok) { message(data.error || "บันทึกไม่สำเร็จ", true); return; }
    $("state").textContent = "มีเส้นแล้ว"; $("state").classList.add("on");
    message("บันทึกแล้ว · กล้องกำลังรีสตาร์ทเพื่อเริ่มนับเข้า/ออก");
    setTimeout(load, 4000);
  });
  $("clear").addEventListener("click", async () => {
    if (!confirm("ลบเส้นประตูของกล้องนี้?")) return;
    const [ok] = await send({ clear: true });
    if (ok) { pts = []; draw(); $("state").textContent = "ยังไม่มีเส้น"; $("state").classList.remove("on"); message("ลบเส้นแล้ว"); }
  });
  load();
})();
