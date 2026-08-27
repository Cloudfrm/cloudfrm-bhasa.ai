/* ime.js — IME-style candidate strip for romanized Nepali (E9).
   - Works per Latin word-run at the caret; protected runs never open a strip.
   - Up to 5 Devanagari candidates + the raw Latin as the last option, always.
   - Digit keys 1–6, arrow keys + Enter, or click select. Space/Enter accept
     the highlighted candidate; Esc keeps the raw Latin.
   - Session memory: an explicit choice ranks first next time (sessionStorage).
   - Visible toggle "रोमन → देवनागरी (चालू) — Ctrl+G"; keyboard-only operable;
     announced to screen readers through the caller's live region. */
(function (root) {
  "use strict";
  const core = root.DeskCore;

  const LABEL = {
    ne: {
      toggle: (on) => `रोमन → देवनागरी (${on ? "चालू" : "बन्द"}) — Ctrl+G`,
      strip: "देवनागरी विकल्प",
      raw: "जस्ताको तस्तै",
      options: (n) => `${core.toDevaDigits(String(n))} विकल्प`,
      chosen: (t) => `${t} छानियो`,
      kept: "जस्ताको तस्तै राखियो",
      on: "रोमन देवनागरी रूपान्तरण चालू",
      off: "रोमन देवनागरी रूपान्तरण बन्द",
    },
    en: {
      toggle: (on) => `Roman → Devanagari (${on ? "on" : "off"}) — Ctrl+G`,
      strip: "Devanagari candidates",
      raw: "keep as typed",
      options: (n) => `${n} options`,
      chosen: (t) => `${t} selected`,
      kept: "kept as typed",
      on: "Roman to Devanagari conversion on",
      off: "Roman to Devanagari conversion off",
    },
  };

  function attach(textarea, opts) {
    const strip = opts.strip;
    const toggleBtn = opts.toggle;
    const lang = () => (opts.lang ? opts.lang() : "ne");
    const announce = opts.announce || function () {};
    const fetchRun = opts.fetchRun; // (fullText, runStart, choices) => Promise<run>
    let enabled = (function () { try { return localStorage.getItem("bhasa-ime") !== "off"; } catch (e) { return true; } })();
    let open = false;
    let items = [];
    let index = 0;
    let run = null;
    let auto = false; // server says this run may convert without an explicit pick
    let req = 0;
    let keptRaw = null; // {text,start} the user chose to keep as typed
    let choices = {};
    try { choices = JSON.parse(sessionStorage.getItem("bhasa-ime-choices") || "{}") || {}; } catch (e) { choices = {}; }

    strip.setAttribute("role", "listbox");
    strip.setAttribute("aria-label", LABEL[lang()].strip);
    strip.hidden = true;
    textarea.setAttribute("aria-autocomplete", "list");
    textarea.setAttribute("aria-controls", strip.id);
    textarea.setAttribute("aria-expanded", "false");

    function saveChoices() { try { sessionStorage.setItem("bhasa-ime-choices", JSON.stringify(choices)); } catch (e) {} }

    function paintToggle() {
      if (!toggleBtn) return;
      toggleBtn.textContent = LABEL[lang()].toggle(enabled);
      toggleBtn.setAttribute("aria-pressed", enabled ? "true" : "false");
      toggleBtn.classList.toggle("on", enabled);
    }

    function setEnabled(on) {
      enabled = on;
      try { localStorage.setItem("bhasa-ime", on ? "on" : "off"); } catch (e) {}
      paintToggle();
      announce(on ? LABEL[lang()].on : LABEL[lang()].off);
      if (!on) close();
    }

    function close() {
      open = false;
      items = [];
      strip.hidden = true;
      strip.innerHTML = "";
      textarea.setAttribute("aria-expanded", "false");
      textarea.removeAttribute("aria-activedescendant");
    }

    function paint() {
      strip.innerHTML = "";
      strip.setAttribute("aria-label", LABEL[lang()].strip);
      items.forEach((item, i) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.id = strip.id + "-opt-" + i;
        btn.setAttribute("role", "option");
        btn.setAttribute("aria-selected", i === index ? "true" : "false");
        btn.className = "ime__opt" + (i === index ? " is-active" : "") + (item.raw ? " is-raw" : "");
        btn.tabIndex = -1;
        const key = document.createElement("kbd");
        key.textContent = String(i + 1);
        const txt = document.createElement("span");
        txt.lang = item.raw ? "en" : "ne";
        txt.textContent = item.text;
        btn.appendChild(key);
        btn.appendChild(txt);
        if (item.raw) {
          const hint = document.createElement("small");
          hint.textContent = LABEL[lang()].raw;
          btn.appendChild(hint);
        }
        btn.addEventListener("mousedown", (e) => e.preventDefault()); // keep focus in textarea
        btn.addEventListener("click", () => accept(i, true));
        strip.appendChild(btn);
      });
      strip.hidden = false;
      open = true;
      textarea.setAttribute("aria-expanded", "true");
      textarea.setAttribute("aria-activedescendant", strip.id + "-opt-" + index);
    }

    function move(delta) {
      if (!open) return;
      index = (index + delta + items.length) % items.length;
      paint();
      announce(items[index].text);
    }

    function accept(i, explicit) {
      if (!open || !run) return null;
      const item = items[i];
      const value = textarea.value;
      const replaced = value.slice(0, run.start) + item.text + value.slice(run.end);
      const caret = run.start + item.text.length;
      textarea.value = replaced;
      textarea.setSelectionRange(caret, caret);
      if (explicit && !item.raw) {
        choices[run.text.toLowerCase()] = item.text;
        saveChoices();
      }
      if (item.raw) keptRaw = { text: run.text, start: run.start };
      announce(item.raw ? LABEL[lang()].kept : LABEL[lang()].chosen(item.text));
      close();
      textarea.dispatchEvent(new Event("ime-accept", { bubbles: true }));
      return item.text;
    }

    function rank(cands, word) {
      const chosen = choices[word.toLowerCase()];
      let list = cands.map((c) => ({ text: c.text, source: c.source }));
      if (chosen) {
        list = [{ text: chosen, source: "session" }].concat(list.filter((c) => c.text !== chosen));
      }
      return list.slice(0, 5);
    }

    /* Ask the server about the run at `found` with the whole composer as
       context (an English-spelled loanword converts on Space only inside a
       Nepali sentence). Not cached: the decision depends on context. */
    async function lookup(found) {
      try { return await fetchRun(textarea.value, found.start, choices); } catch (e) { return null; }
    }

    function usable(decision) {
      return decision && decision.parses && decision.candidates && decision.candidates.length;
    }

    function openFor(found, decision) {
      run = found;
      auto = !!decision.auto;
      items = rank(decision.candidates, found.text).concat([{ text: found.text, raw: true }]);
      index = 0;
    }

    async function refresh() {
      if (!enabled) { close(); return; }
      const caret = textarea.selectionStart;
      const found = core.runAtCaret(textarea.value, caret);
      if (!found) { close(); run = null; return; }
      if (keptRaw && keptRaw.text === found.text && keptRaw.start === found.start) { close(); return; }
      if (core.protectedReason(found.text)) { close(); run = null; return; }
      const my = ++req;
      const decision = await lookup(found);
      if (my !== req) return;
      if (!usable(decision)) { close(); run = null; return; }
      openFor(found, decision);
      paint();
      announce(LABEL[lang()].options(items.length) + ": " + items.map((it, i) => `${i + 1} ${it.text}`).join(", "));
    }

    /* Space/Enter with the default highlighted: convert only if the server
       said this run may auto-convert; otherwise keep the raw Latin. */
    function acceptDefault() {
      if (index === 0 && !auto) return accept(items.length - 1, false);
      return accept(index, index !== 0);
    }

    let timer = null;
    textarea.addEventListener("input", () => {
      if (keptRaw) {
        const f = core.runAtCaret(textarea.value, textarea.selectionStart);
        if (!f || f.text !== keptRaw.text || f.start !== keptRaw.start) keptRaw = null;
      }
      clearTimeout(timer);
      timer = setTimeout(refresh, 60);
    });
    textarea.addEventListener("click", () => { clearTimeout(timer); timer = setTimeout(refresh, 60); });
    textarea.addEventListener("blur", () => setTimeout(close, 120));

    textarea.addEventListener("keydown", async (event) => {
      if (event.ctrlKey && (event.key === "g" || event.key === "G")) {
        event.preventDefault();
        setEnabled(!enabled);
        return;
      }
      if (!enabled) return;
      if (open) {
        if (/^[1-6]$/.test(event.key)) {
          const i = Number(event.key) - 1;
          if (i < items.length) { event.preventDefault(); accept(i, true); }
          return;
        }
        if (event.key === "ArrowRight" || event.key === "ArrowDown") { event.preventDefault(); move(1); return; }
        if (event.key === "ArrowLeft" || event.key === "ArrowUp") { event.preventDefault(); move(-1); return; }
        if (event.key === "Escape") { event.preventDefault(); accept(items.length - 1, false); return; }
        if (event.key === "Enter" && !event.shiftKey) {
          // Enter accepts the highlighted candidate; it does NOT send.
          event.preventDefault();
          event.stopImmediatePropagation();
          acceptDefault();
          return;
        }
        if (event.key === " ") {
          event.preventDefault();
          acceptDefault();
          insertAtCaret(" ");
          return;
        }
        if (event.key === "Tab") { acceptDefault(); return; }
      } else if (event.key === " " || (event.key === "Enter" && !event.shiftKey)) {
        // Fast typists: the strip may not have opened yet. Resolve the run
        // before the space/enter lands.
        const found = core.runAtCaret(textarea.value, textarea.selectionStart);
        if (!found || core.protectedReason(found.text)) return;
        if (keptRaw && keptRaw.text === found.text && keptRaw.start === found.start) return;
        event.preventDefault();
        if (event.key === "Enter") event.stopImmediatePropagation();
        clearTimeout(timer);
        const decision = await lookup(found);
        if (usable(decision) && decision.auto) {
          openFor(found, decision);
          open = true;
          accept(0, false);
        }
        if (event.key === " ") insertAtCaret(" ");
        else textarea.dispatchEvent(new CustomEvent("ime-submit", { bubbles: true }));
      }
    }, true);

    function insertAtCaret(text) {
      const s = textarea.selectionStart, e = textarea.selectionEnd;
      textarea.value = textarea.value.slice(0, s) + text + textarea.value.slice(e);
      textarea.setSelectionRange(s + text.length, s + text.length);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    }

    if (toggleBtn) toggleBtn.addEventListener("click", () => setEnabled(!enabled));
    paintToggle();

    return {
      get enabled() { return enabled; },
      get open() { return open; },
      setEnabled,
      paintToggle,
      close,
      /* Resolve any pending run at the caret before submit. */
      async commitPending() {
        if (open) { acceptDefault(); return; }
        if (!enabled) return;
        const found = core.runAtCaret(textarea.value, textarea.selectionStart);
        if (!found || core.protectedReason(found.text)) return;
        if (keptRaw && keptRaw.text === found.text && keptRaw.start === found.start) return;
        const decision = await lookup(found);
        if (usable(decision) && decision.auto) {
          openFor(found, decision);
          open = true;
          accept(0, false);
        }
      },
    };
  }

  root.DeskIME = { attach, LABEL };
})(window);
