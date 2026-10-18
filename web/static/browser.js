// "Camera on my computer": open a local camera in the browser, send frames to
// the server for detection, draw the returned boxes over the live picture.
//
// Resolution independence: the camera can be any size. We rotate + shrink each
// frame so its long side is at most `long_side` before sending, and the server
// answers with boxes as 0..1 fractions, which we scale to whatever size the
// picture is shown at.
(function () {
  const $ = (id) => document.getElementById(id);
  const P = window.PORTAL;
  const MAX_FPS = P.maxFps || 10;  // frames sent per second at most (ตั้งค่าระบบ)
  const JPEG_QUALITY = 0.8;
  const DISPLAY_LONG = 1280;   // size of the on-screen canvas (long side)

  const video = document.createElement("video");
  video.muted = true; video.playsInline = true;
  const view = $("view"), ctx = view.getContext("2d");
  const send = document.createElement("canvas"), sctx = send.getContext("2d");

  let stream = null, running = false, latest = null, sessionId = newId();
  let frames = 0, fpsSince = performance.now();

  function newId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return "s" + Math.random().toString(36).slice(2) + Date.now().toString(36);
  }
  function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };

  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("insecure").hidden = false;
    $("find").disabled = $("start").disabled = true;
    return;
  }

  // ---------------------------------------------------------------- cameras
  async function findCameras() {
    showError("");
    try {
      // Ask once so the browser reveals camera names.
      const probe = await navigator.mediaDevices.getUserMedia({ video: true });
      probe.getTracks().forEach((t) => t.stop());
    } catch (e) {
      showError("เปิดกล้องไม่ได้: " + (e.name === "NotAllowedError" ? "ไม่ได้รับอนุญาตให้ใช้กล้อง" : e.message));
      return;
    }
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const select = $("device");
    select.innerHTML = devices.length ? "" : '<option value="">ไม่พบกล้อง</option>';
    devices.forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId; opt.textContent = d.label || `กล้อง ${i + 1}`;
      select.appendChild(opt);
    });
    const saved = store.get("portal.camera");
    if (saved && devices.some((d) => d.deviceId === saved)) select.value = saved;
  }

  async function start() {
    showError("");
    if (!$("device").value) await findCameras();
    const deviceId = $("device").value;
    if (!deviceId) return;
    store.set("portal.camera", deviceId);
    try {
      // Ask for a high resolution; whatever the camera really gives is fine.
      stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      });
    } catch (e) {
      showError("เปิดกล้องไม่ได้: " + e.message);
      return;
    }
    video.srcObject = stream;
    await video.play();
    sessionId = newId();
    running = true; latest = null;
    $("placeholder").hidden = true; view.hidden = false;
    $("start").disabled = true; $("stop").disabled = false; $("device").disabled = true;
    $("st-camera").textContent = `${video.videoWidth}×${video.videoHeight}`;
    requestAnimationFrame(render);
    loop();
  }

  function stop() {
    running = false;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    closeSession();
    $("start").disabled = false; $("stop").disabled = true; $("device").disabled = false;
  }

  function closeSession() {
    const body = new FormData();
    body.append("csrfmiddlewaretoken", P.csrf);
    navigator.sendBeacon(`${P.closeUrl}?session=${encodeURIComponent(sessionId)}`, body);
  }

  // ------------------------------------------------------------- geometry
  // Draw the video rotated (clockwise) onto a canvas of the given long side.
  function drawRotated(target, tctx, longSide) {
    const rot = parseInt($("rotate").value, 10);
    const vw = video.videoWidth, vh = video.videoHeight;
    const sideways = rot === 90 || rot === 270;
    const rw = sideways ? vh : vw, rh = sideways ? vw : vh;     // rotated size
    const scale = Math.min(1, longSide / Math.max(rw, rh));      // never enlarge
    const w = Math.round(rw * scale), h = Math.round(rh * scale);
    if (target.width !== w || target.height !== h) { target.width = w; target.height = h; }
    tctx.save();
    tctx.translate(w / 2, h / 2);
    tctx.rotate((rot * Math.PI) / 180);
    const dw = (sideways ? h : w), dh = (sideways ? w : h);
    tctx.drawImage(video, -dw / 2, -dh / 2, dw, dh);
    tctx.restore();
    return { w, h };
  }

  // ------------------------------------------------------- send / receive
  function options() {
    return new URLSearchParams({
      session: sessionId,
      rotate: "0", // already rotated here, in the browser
      long_side: $("long_side").value,
      face: $("face").checked ? "1" : "0",
      vlm: $("vlm").checked ? "1" : "0",
    });
  }

  async function loop() {
    while (running) {
      const t0 = performance.now();
      if (video.readyState >= 2) {
        const { w, h } = drawRotated(send, sctx, parseInt($("long_side").value, 10));
        $("st-sent").textContent = `${w}×${h}`;
        const blob = await new Promise((r) => send.toBlob(r, "image/jpeg", JPEG_QUALITY));
        try {
          const res = await fetch(`${P.frameUrl}?${options()}`, {
            method: "POST", body: blob,
            headers: { "X-CSRFToken": P.csrf, "Content-Type": "image/jpeg" },
          });
          if (res.redirected) { location.reload(); return; } // session expired -> login
          const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
          if (!res.ok) {
            showError(data.error || `เซิร์ฟเวอร์ตอบ ${res.status}`);
            if (res.status === 429 || res.status === 400) { stop(); return; }
          } else {
            showError("");
            latest = data;
            updateStats(data, performance.now() - t0);
          }
        } catch (e) {
          showError("ติดต่อเซิร์ฟเวอร์ไม่ได้ กำลังลองใหม่…");
          await sleep(1000);
        }
      }
      const wait = 1000 / MAX_FPS - (performance.now() - t0);
      if (wait > 0) await sleep(wait);
    }
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function updateStats(data, roundTrip) {
    frames += 1;
    const now = performance.now();
    if (now - fpsSince > 1000) {
      $("st-fps").textContent = (frames * 1000 / (now - fpsSince)).toFixed(1) + " fps";
      frames = 0; fpsSince = now;
    }
    $("st-ms").textContent = `${data.ms} ms (ไป-กลับ ${Math.round(roundTrip)} ms)`;
    $("st-people").textContent = data.count;
    const list = $("people");
    list.innerHTML = data.people.length ? "" : '<li class="muted">ไม่มีคน</li>';
    data.people.forEach((p) => {
      const li = document.createElement("li");
      const b = document.createElement("b"); b.textContent = p.name || (p.track_id >= 0 ? `ID ${p.track_id}` : "person");
      li.appendChild(b);
      const detail = [p.clothing && `เสื้อ/กางเกง ${p.clothing}`, p.attributes].filter(Boolean).join(" · ");
      if (detail) { li.appendChild(document.createElement("br")); li.appendChild(document.createTextNode(detail)); }
      list.appendChild(li);
    });
  }

  // -------------------------------------------------------------- drawing
  function render() {
    if (!running) return;
    if (video.readyState >= 2) {
      const { w, h } = drawRotated(view, ctx, DISPLAY_LONG);
      if (latest) drawResults(latest, w, h);
    }
    requestAnimationFrame(render);
  }

  function drawResults(data, w, h) {
    const line = Math.max(2, w / 400), font = Math.max(12, Math.round(w / 55));
    ctx.font = `${font}px "Segoe UI", "Leelawadee UI", sans-serif`;
    ctx.textBaseline = "bottom";
    data.faces.forEach(([x1, y1, x2, y2]) => {
      ctx.strokeStyle = "#19b35a"; ctx.lineWidth = line;
      ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
    });
    data.people.forEach((p) => {
      const [x1, y1, x2, y2] = p.box;
      const x = x1 * w, y = y1 * h;
      ctx.strokeStyle = "#e0249a"; ctx.lineWidth = line;
      ctx.strokeRect(x, y, (x2 - x1) * w, (y2 - y1) * h);
      label(p.label, x, Math.max(y, font + 6), "#e0249a", font);
      if (p.attributes) label(p.attributes, x, Math.min(y2 * h, h - 2), "#14465a", Math.round(font * 0.85));
    });
  }

  function label(text, x, y, bg, font) {
    ctx.font = `${font}px "Segoe UI", "Leelawadee UI", sans-serif`;
    const pad = 4, tw = ctx.measureText(text).width;
    ctx.fillStyle = bg; ctx.fillRect(x, y - font - pad * 2, tw + pad * 2, font + pad * 2);
    ctx.fillStyle = "#fff"; ctx.fillText(text, x + pad, y - pad);
  }

  // ---------------------------------------------------------------- wiring
  $("find").addEventListener("click", findCameras);
  $("start").addEventListener("click", start);
  $("stop").addEventListener("click", stop);
  ["face", "vlm", "long_side"].forEach((id) => $(id).addEventListener("change", () => { latest = null; }));
  window.addEventListener("pagehide", () => { if (running) closeSession(); });
})();
