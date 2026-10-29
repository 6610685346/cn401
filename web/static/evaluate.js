// "ทดสอบด้วยวิดีโอ": upload with a progress bar, live progress, and the review page.
(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const csrf = (window.EVAL && window.EVAL.csrf) || ($('input[name="csrfmiddlewaretoken"]') || {}).value || "";

  // ------------------------------------------------------------ upload page
  const form = $("#upload-form");
  if (form) {
    const input = $("#video"), drop = $("#drop"), picked = $("#picked");
    const show = () => {
      const f = input.files[0];
      picked.textContent = f ? `${f.name} · ${(f.size / 1048576).toFixed(1)} MB` : "";
      const name = $("#name");
      if (f && !name.value) name.value = f.name.replace(/\.[^.]+$/, "");
    };
    input.addEventListener("change", show);
    ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, () => drop.classList.remove("over")));
    drop.addEventListener("drop", (ev) => {
      ev.preventDefault();
      if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; show(); }
    });
    // XHR so a big video shows how far the upload is
    form.addEventListener("submit", (ev) => {
      if (!input.files.length) return;
      ev.preventDefault();
      const bar = $("#upload-progress"), fill = bar.firstElementChild, btn = $("#upload-btn");
      bar.hidden = false; btn.disabled = true;
      const xhr = new XMLHttpRequest();
      xhr.open("POST", form.action || location.href);
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) fill.style.width = `${(100 * e.loaded / e.total).toFixed(1)}%`; };
      xhr.onload = () => { location.href = xhr.responseURL || location.href; };
      xhr.onerror = () => { btn.disabled = false; bar.hidden = true; alert("อัปโหลดไม่สำเร็จ ลองใหม่อีกครั้ง"); };
      xhr.send(new FormData(form));
    });
  }

  // list page: refresh rows that are still running
  const STATUS_TEXT = { queued: "รอคิว", running: "กำลังประมวลผล", done: "เสร็จ", error: "ผิดพลาด", cancelled: "ยกเลิก" };
  const live = $$("tr[data-video]").filter((tr) => ["queued", "running"].includes(tr.dataset.status));
  if (live.length) {
    const tick = async () => {
      for (const tr of live) {
        try {
          const data = await (await fetch(tr.dataset.statusUrl, { cache: "no-store" })).json();
          if (data.status !== tr.dataset.status && !["queued", "running"].includes(data.status)) { location.reload(); return; }
          tr.dataset.status = data.status;
          $('[data-field="status"]', tr).textContent = STATUS_TEXT[data.status] || data.status;
          $('[data-field="events"]', tr).textContent = data.events;
          const p = $(".progress > div", tr); if (p) p.style.width = `${(data.progress * 100).toFixed(0)}%`;
        } catch (e) { /* server busy */ }
      }
    };
    setInterval(tick, 2000);
  }

  // ------------------------------------------------------------ detail page
  const run = $("#run");
  if (run && run.dataset.running === "1") {
    const img = $("#run-preview"), empty = $("#run-empty");
    const tick = async () => {
      try {
        const data = await (await fetch(run.dataset.statusUrl, { cache: "no-store" })).json();
        if (!["queued", "running"].includes(data.status)) { location.reload(); return; }
        $("#run-bar").style.width = `${(data.progress * 100).toFixed(1)}%`;
        $("#run-message").textContent = data.message || "";
        $("#run-events").textContent = data.events;
        if (data.has_preview) {
          img.src = `${img.dataset.src}?t=${Date.now()}`;
          img.hidden = false; empty.hidden = true;
        }
      } catch (e) { /* server busy */ }
    };
    tick();
    setInterval(tick, 1500);
  }

  const metrics = $("#metrics");
  let refreshTimer = null;
  const refreshMetrics = () => {
    if (!metrics) return;
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(async () => {
      try { metrics.innerHTML = await (await fetch(metrics.dataset.url, { cache: "no-store" })).text(); } catch (e) { /* ignore */ }
    }, 250);
  };

  async function save(card, data) {
    card.classList.add("saving");
    try {
      const res = await fetch(card.dataset.reviewUrl, {
        method: "POST", headers: { "Content-Type": "application/json", "X-CSRFToken": csrf }, body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error(res.status);
      refreshMetrics();
      return true;
    } catch (e) {
      alert("บันทึกไม่สำเร็จ ลองอีกครั้ง");
      return false;
    } finally {
      card.classList.remove("saving");
    }
  }

  function paint(card) {
    const field = card.dataset.kind === "sighting" ? "gt_status" : "gt_direction";
    const on = $(`.seg[data-field="${field}"] button.on`, card);
    const value = on ? on.dataset.value : "";
    card.classList.toggle("is-ok", value === "ok");
    card.classList.toggle("is-bad", !!value && value !== "ok");
    card.dataset.done = value ? "1" : "0";
  }

  $$(".review").forEach((card) => {
    $$(".seg button", card).forEach((btn) => btn.addEventListener("click", async () => {
      const seg = btn.parentElement, field = seg.dataset.field;
      const value = btn.classList.contains("on") ? "" : btn.dataset.value; // click again = undo
      const payload = { [field]: value };
      if (field === "gt_status" && value !== "ok") payload.gt_identity = "";
      if (!(await save(card, payload))) return;
      $$("button", seg).forEach((b) => b.classList.toggle("on", b === btn && value !== ""));
      if (payload.gt_identity === "") { const sel = $("select.identity", card); if (sel) sel.value = ""; }
      paint(card);
    }));
    const sel = $("select.identity", card);
    if (sel) sel.addEventListener("change", async () => {
      const payload = { gt_identity: sel.value };
      const okBtn = $('.seg[data-field="gt_status"] button[data-value="ok"]', card);
      const anyOn = $('.seg[data-field="gt_status"] button.on', card);
      if (sel.value && !anyOn) payload.gt_status = "ok"; // naming someone means it is a real person
      if (!(await save(card, payload))) return;
      if (payload.gt_status) { okBtn.classList.add("on"); paint(card); }
    });
  });

  // filters
  $$("#filters .chip-btn").forEach((chip) => chip.addEventListener("click", () => {
    $$("#filters .chip-btn").forEach((c) => c.classList.toggle("on", c === chip));
    const f = chip.dataset.filter;
    $$(".review").forEach((card) => {
      const kind = card.dataset.kind;
      card.hidden = !(f === "all" || (f === "todo" && card.dataset.done !== "1")
        || (f === "sighting" && kind === "sighting") || (f === "cross" && kind !== "sighting"));
    });
  }));

  // missed counts: save without leaving the page
  const missed = $("#missed-form");
  if (missed) missed.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const res = await fetch(missed.action, { method: "POST", body: new FormData(missed), headers: { "x-requested-with": "fetch" } });
    $("#missed-saved").textContent = res.ok ? "บันทึกแล้ว" : "บันทึกไม่สำเร็จ";
    refreshMetrics();
  });
})();
