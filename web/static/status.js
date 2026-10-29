// Polls /api/status/ and fills in every [data-camera] block on the page.
(function () {
  const STATUS_TEXT = {
    running: "กำลังทำงาน", stopped: "หยุด", starting: "กำลังเริ่ม", loading: "โหลดโมเดล",
    connecting: "กำลังเชื่อมต่อ", reconnecting: "กำลังเชื่อมต่อใหม่", error: "ผิดพลาด",
  };
  const COLORS = {
    black: "#111", white: "#fff", gray: "#888", red: "#d22", orange: "#f80", yellow: "#ec0",
    green: "#2a2", cyan: "#0cc", blue: "#24d", purple: "#82c", pink: "#f6b", brown: "#843", coral: "#f76",
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const size = (s) => (s ? `${s[0]}×${s[1]}` : "-");
  const chip = (c) => `<span class="chip" style="background:${COLORS[c] || "transparent"}"></span> ${esc(c)}`;

  const img = document.getElementById("live");
  let lastStatus = null;
  function connectImage() {
    if (!img) return;
    img.src = img.dataset.src + "?t=" + Date.now(); // fresh connection every time
  }
  if (img) {
    img.addEventListener("error", () => setTimeout(connectImage, 3000));
    connectImage();
  }

  function fill(block, cam) {
    const set = (field, html) => {
      const el = block.querySelector(`[data-field="${field}"]`);
      if (el) el.innerHTML = html;
    };
    const status = cam.status || "stopped";
    const pill = block.querySelector('[data-field="status"]');
    if (pill) { pill.className = "pill s-" + status; pill.textContent = STATUS_TEXT[status] || status; }
    set("message", esc(cam.message || ""));
    set("people_now", cam.status === "running" ? cam.people_now : "-");
    set("fps", cam.status === "running" ? cam.fps : "-");
    set("source_size", size(cam.source_size));
    set("process_size", size(cam.process_size));
    const perf = block.querySelector('[data-field="perf"]');
    if (perf && cam.perf) {
      const p = cam.perf, pct = (v) => `${Math.round(v * 100)}%`;
      perf.innerHTML = `
        <div class="stat">YOLO ทำงาน <b>${pct(p.detect_share)}</b> <span class="small">ของเฟรม · พัก ${p.idle_skips} เฟรม</span></div>
        <div class="stat">หาใบหน้า <b>${pct(p.face_share)}</b> <span class="small">ของเฟรม · ${p.face_regions} บริเวณ</span></div>
        <div class="stat">จำหน้า <b>${p.face_embeddings}</b> <span class="small">ครั้ง</span></div>
        <div class="stat">ไม่บันทึกซ้ำ <b>${p.rows_skipped}</b> <span class="small">แถว</span></div>`;
      const badge = block.querySelector('[data-field="backend"]');
      if (badge) {
        const faceNames = { sface: "SFace", arcface_mbf: "ArcFace-MBF", arcface_r50: "ArcFace-R50" };
        const parts = [p.person_model || "YOLO", p.backend === "openvino" ? "OpenVINO" : "PyTorch"];
        if (p.face_model) parts.push(faceNames[p.face_model] || p.face_model);
        if (p.reid) parts.push("ReID");
        if (p.full_res) parts.push("full-res");
        badge.textContent = parts.join(" · ");
      }
    }
    const feed = block.querySelector('[data-field="recent"]');
    if (feed && cam.recent) {
      feed.innerHTML = cam.recent.length
        ? cam.recent.map((r) => `<li><div class="person-cell"><span class="mini-avatar${r.name && r.name !== "unknown" ? "" : " unknown"}">${esc((r.name && r.name !== "unknown" ? r.name : "?").slice(0, 1))}</span>
             <div><b>${r.global_id ? "G" + esc(r.global_id) + " " : ""}${esc(r.name)}</b> <span class="muted small">#${esc(r.track_id)} · ${esc(r.time.slice(11))}</span><br>
             <span class="small">เสื้อ ${chip(r.shirt_color)} · กางเกง ${chip(r.pants_color)}</span></div></div></li>`).join("")
        : '<li class="muted">ยังไม่มี</li>';
    }
    if (img && block.contains(img)) {
      if (status === "running" && lastStatus !== "running") connectImage(); // camera came back
      lastStatus = status;
    }
  }

  async function poll() {
    try {
      const res = await fetch(window.STATUS_URL, { cache: "no-store" });
      if (res.redirected || !res.ok) return; // logged out
      const data = await res.json();
      document.querySelectorAll("[data-camera]").forEach((block) => {
        const cam = data.cameras[block.dataset.camera];
        if (cam) fill(block, cam);
      });
      // dashboard totals
      const cams = Object.values(data.cameras || {});
      const running = cams.filter((c) => c.status === "running");
      const total = (name, value) => document.querySelectorAll(`[data-total="${name}"]`).forEach((el) => { el.textContent = value; });
      total("running", `${running.length} / ${cams.length}`);
      total("people", running.reduce((sum, c) => sum + (Number(c.people_now) || 0), 0));
    } catch (e) { /* server restarting; try again next tick */ }
  }
  poll();
  setInterval(poll, 2000);
})();
