/* desk-ui.js — DOM renderers for every thread / composer / header state.
   Pure functions of data → element; no app state. Used by desk.js and by the
   dev harness (states.html) so every state can be rendered in isolation.
   Rules: a refusal is NOT an error (neutral surface, same geometry as an
   answer); the alert palette is only for genuine breakage; no state is
   conveyed by colour alone; quoted evidence is rendered verbatim with its
   own lang attribute and is never reformatted. */
(function (root) {
  "use strict";
  const core = root.DeskCore;

  const COPY = {
    ne: {
      agent: "bhasa",
      member: "सदस्यको प्रश्न",
      quoted: "दस्तावेजबाट उद्धृत",
      source: "स्रोत",
      langName: { ne: "नेपाली", en: "English" },
      refusalLabel: {
        general: "लोड गरिएका दस्तावेजमा यो प्रश्नको जवाफ भेटिएन",
        quantity: "मागिएको अंक लोड गरिएका दस्तावेजमा छैन",
      },
      refusalNote: "bhasa ले दस्तावेजमा नभएको कुरा लेख्दैन। तल उत्पादनको स्थिर अस्वीकार वाक्य जस्ताको तस्तै छ।",
      refusalFrozen: "स्थिर अस्वीकार वाक्य",
      openTicket: "टिकट खोल्नुहोस्",
      ticketOpened: "टिकट खुल्यो",
      searching: "दस्तावेज खोज्दै…",
      searchingLong: "अझै खोज्दै छ — सामान्यभन्दा ढिलो भयो। पर्खनुहोस् वा रद्द गर्नुहोस्।",
      cancel: "रद्द",
      failed: "पठाउन सकिएन",
      failedBody: "सेवाबाट जवाफ आएन। सन्देश यहीँ सुरक्षित छ — फेरि पठाउन सक्नुहुन्छ।",
      timeout: "समय सकियो",
      timeoutBody: "३० सेकेन्डसम्म जवाफ आएन। सन्देश यहीँ छ — फेरि पठाउन सक्नुहुन्छ।",
      retry: "फेरि पठाउनुहोस्",
      rateLimited: "अहिले धेरै अनुरोध भयो",
      rateLimitedBody: "एकछिन पर्खेर फेरि पठाउनुहोस्।",
      rateLimitedAfter: (s) => `${core.toDevaDigits(String(s))} सेकेन्डपछि फेरि पठाउन सकिन्छ।`,
      offline: "इन्टरनेट जडान छैन",
      offlineBody: "जडान फर्केपछि पठाउन सकिन्छ। इतिहास र लेख्ने ठाउँ खुला छन्।",
      online: "जडान फर्कियो",
      serviceDown: "सेवा पुग्न सकिएन",
      empty: "नयाँ संवाद",
      emptyBody: "सदस्यले सोधेको प्रश्न जस्ताको तस्तै लेख्नुहोस्, वा तलको नमूना प्रश्न छान्नुहोस्। bhasa ले लोड गरिएका दस्तावेजबाट मिल्ने अंश उद्धृत गर्छ; नभेटिए स्पष्ट भन्छ।",
      credWarn: "यो सन्देशमा पिन, ओटीपी, कार्ड नम्बर वा पासवर्ड जस्तो देखिने कुरा छ। पठाउनुअघि हटाउनुहोस् — bhasa ले ती कहिल्यै माग्दैन।",
      credDecline: "गोप्य विवरण भएकाले प्रशोधन गरिएन",
      credRule: "पिन, पासवर्ड, ओटीपी वा सीभीभी यहाँ कहिल्यै नलेख्नुहोस्।",
      fontWarn: "देवनागरी फन्ट लोड भएन — अक्षर बिग्रिएर देखिन सक्छ।",
      micFail: "माइक्रोफोन चलेन। लेखेर पठाउनुहोस्।",
      micBlocked: "ब्राउजरले माइक्रोफोन रोकेको छ। ठेगाना पट्टीबाट अनुमति दिनुहोस्, वा लेख्नुहोस्।",
      unsupported: "यो ब्राउजरमा आवाज समर्थित छैन। लेखेर पठाउनुहोस्।",
      warn: "सावधान",
      error: "त्रुटि",
      info: "जानकारी",
    },
    en: {
      agent: "bhasa",
      member: "Member's question",
      quoted: "Quoted from a document",
      source: "Source",
      langName: { ne: "नेपाली", en: "English" },
      refusalLabel: {
        general: "No loaded document answers this question",
        quantity: "The requested figure is not in the loaded documents",
      },
      refusalNote: "bhasa does not write what the documents do not say. Below is the product's fixed refusal sentence, verbatim (it is Nepali; it is never translated).",
      refusalFrozen: "fixed refusal sentence",
      openTicket: "Open a ticket",
      ticketOpened: "Ticket opened",
      searching: "Searching the documents…",
      searchingLong: "Still searching — slower than usual. Wait, or cancel.",
      cancel: "Cancel",
      failed: "Could not send",
      failedBody: "No reply from the service. The message is kept here — you can send it again.",
      timeout: "Timed out",
      timeoutBody: "No reply within 30 seconds. The message is kept here — you can send it again.",
      retry: "Send again",
      rateLimited: "Too many requests right now",
      rateLimitedBody: "Wait a moment, then send again.",
      rateLimitedAfter: (s) => `You can send again in ${s} seconds.`,
      offline: "No internet connection",
      offlineBody: "Sending resumes when the connection returns. History and the composer stay open.",
      online: "Connection restored",
      serviceDown: "Service unreachable",
      empty: "New conversation",
      emptyBody: "Type the member's question exactly as asked, or pick a sample question below. bhasa quotes the matching passage from the loaded documents; if nothing matches, it says so.",
      credWarn: "This message contains something shaped like a PIN, OTP, card number, or password. Remove it before sending — bhasa never asks for these.",
      credDecline: "Not processed: it contained a credential",
      credRule: "Never type a PIN, password, OTP, or CVV here.",
      fontWarn: "The Devanagari font did not load — glyphs may render broken.",
      micFail: "The microphone did not work. Type instead.",
      micBlocked: "The browser is blocking the microphone. Allow it from the address bar, or type.",
      unsupported: "Voice is not supported in this browser. Type instead.",
      warn: "Warning",
      error: "Error",
      info: "Note",
    },
  };

  function copy(lang) { return COPY[lang === "en" ? "en" : "ne"]; }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const child of [].concat(children || [])) {
      if (child === null || child === undefined || child === false) continue;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  }

  /* ---------------------------------------------------------- turns */

  function memberTurn({ text, lang, pending, redacted }) {
    const c = copy(lang);
    const wrap = el("div", { class: "turn member", "data-state": pending ? "pending" : "sent", lang: lang === "en" ? "en" : "ne" });
    wrap.appendChild(el("div", { class: "who", text: c.member + (redacted ? " · " + c.credDecline : "") }));
    wrap.appendChild(el("div", { class: "bubble", text: text }));
    return wrap;
  }

  function answerCard({ reply, passage, question_language }) {
    const lang = question_language === "en" ? "en" : "ne";
    const c = copy(lang);
    const plang = passage && passage.language === "en" ? "en" : "ne";
    const wrap = el("div", { class: "turn agent", "data-kind": "answer", lang });
    wrap.appendChild(el("div", { class: "who", text: c.agent }));
    const card = el("div", { class: "card answer", role: "group", "aria-label": c.quoted });
    card.appendChild(el("div", { class: "card__label" }, [
      el("span", { class: "card__glyph", "aria-hidden": "true", text: "❝" }),
      el("span", { text: c.quoted + (passage && passage.title ? " · " + passage.title : "") }),
    ]));
    // Verbatim. Own lang so Devanagari line-height applies; never reformatted.
    card.appendChild(el("blockquote", { class: "passage", lang: plang, text: reply }));
    if (passage) {
      card.appendChild(el("div", { class: "card__meta", text: `${c.source}: ${passage.source} · ${passage.id} · ${c.langName[plang]}` }));
    }
    wrap.appendChild(card);
    return wrap;
  }

  function refusalCard({ reply, refusal_type, question_language }, handlers) {
    const lang = question_language === "en" ? "en" : "ne";
    const c = copy(lang);
    const type = refusal_type === "quantity" ? "quantity" : "general";
    const wrap = el("div", { class: "turn agent", "data-kind": "refusal", lang });
    wrap.appendChild(el("div", { class: "who", text: c.agent }));
    // Neutral surface: same geometry and seriousness as an answer. No alert palette.
    const card = el("div", { class: "card refusal", role: "group", "aria-label": c.refusalLabel[type] });
    card.appendChild(el("div", { class: "card__label" }, [
      el("span", { class: "card__glyph", "aria-hidden": "true", text: "▢" }),
      el("span", { text: c.refusalLabel[type] }),
    ]));
    card.appendChild(el("p", { class: "card__note", text: c.refusalNote }));
    card.appendChild(el("blockquote", { class: "passage frozen", lang: "ne", "aria-label": c.refusalFrozen, text: reply }));
    if (handlers && handlers.onTicket) {
      const btn = el("button", { type: "button", class: "btn ghost", text: c.openTicket, onclick: () => handlers.onTicket(btn) });
      card.appendChild(el("div", { class: "card__actions" }, btn));
    }
    wrap.appendChild(card);
    return wrap;
  }

  function credentialDeclineCard({ reply, language }) {
    const lang = language === "en" ? "en" : "ne";
    const c = copy(lang);
    const wrap = el("div", { class: "turn agent", "data-kind": "credential_decline", lang });
    wrap.appendChild(el("div", { class: "who", text: c.agent }));
    const card = el("div", { class: "card refusal", role: "group", "aria-label": c.credDecline });
    card.appendChild(el("div", { class: "card__label" }, [
      el("span", { class: "card__glyph", "aria-hidden": "true", text: "▢" }),
      el("span", { text: c.credDecline }),
    ]));
    card.appendChild(el("p", { class: "passage", lang, text: reply }));
    wrap.appendChild(card);
    return wrap;
  }

  /* --------------------------------------------------- progress / errors */

  function searching(lang) {
    const c = copy(lang);
    // Quiet progress: one status line, no typing dots (the API returns one buffered response).
    const node = el("div", { class: "status-line", role: "status", "aria-live": "polite", "data-state": "searching", lang });
    node.appendChild(el("span", { class: "status-line__glyph", "aria-hidden": "true", text: "◌" }));
    node.appendChild(el("span", { class: "status-line__text", text: c.searching }));
    return node;
  }

  function escalateSearching(node, lang, onCancel) {
    const c = copy(lang);
    node.setAttribute("data-state", "searching-long");
    node.querySelector(".status-line__text").textContent = c.searchingLong;
    if (onCancel && !node.querySelector("button")) {
      node.appendChild(el("button", { type: "button", class: "btn tiny ghost", text: c.cancel, onclick: onCancel }));
    }
    return node;
  }

  function failureCard({ lang, reason, onRetry }) {
    const c = copy(lang);
    const timeout = reason === "timeout";
    // Genuine breakage: alert palette allowed, but never colour alone — glyph + words.
    const wrap = el("div", { class: "turn agent", "data-kind": "failure", lang });
    const card = el("div", { class: "card error", role: "alert" });
    card.appendChild(el("div", { class: "card__label" }, [
      el("span", { class: "card__glyph", "aria-hidden": "true", text: "⚠" }),
      el("span", { text: (timeout ? c.timeout : c.failed) + " · " + c.error }),
    ]));
    card.appendChild(el("p", { class: "card__note", text: timeout ? c.timeoutBody : c.failedBody }));
    if (onRetry) card.appendChild(el("div", { class: "card__actions" }, el("button", { type: "button", class: "btn", text: c.retry, onclick: onRetry })));
    wrap.appendChild(card);
    return wrap;
  }

  function rateLimitCard({ lang, retryAfter, onRetry }) {
    const c = copy(lang);
    const wrap = el("div", { class: "turn agent", "data-kind": "rate-limit", lang });
    const card = el("div", { class: "card wait", role: "status" });
    card.appendChild(el("div", { class: "card__label" }, [
      el("span", { class: "card__glyph", "aria-hidden": "true", text: "⏸" }),
      el("span", { text: c.rateLimited }),
    ]));
    // No invented number: a countdown appears only when the server sent Retry-After.
    card.appendChild(el("p", { class: "card__note", text: retryAfter ? c.rateLimitedAfter(retryAfter) : c.rateLimitedBody }));
    if (onRetry) card.appendChild(el("div", { class: "card__actions" }, el("button", { type: "button", class: "btn ghost", text: c.retry, onclick: onRetry })));
    wrap.appendChild(card);
    return wrap;
  }

  function offlineBanner(lang, kind) {
    const c = copy(lang);
    const down = kind === "service";
    const node = el("div", { class: "banner", role: "status", "aria-live": "polite", "data-state": down ? "service-down" : "offline", lang });
    node.appendChild(el("span", { class: "banner__glyph", "aria-hidden": "true", text: "⚠" }));
    node.appendChild(el("strong", { text: down ? c.serviceDown : c.offline }));
    node.appendChild(el("span", { text: " · " + c.offlineBody }));
    return node;
  }

  function emptyThread(lang) {
    const c = copy(lang);
    const node = el("div", { class: "empty-thread", "data-state": "empty", lang });
    node.appendChild(el("h2", { text: c.empty }));
    node.appendChild(el("p", { text: c.emptyBody }));
    return node;
  }

  /* ---------------------------------------------------- composer notices */

  function composerNotice({ kind, lang }) {
    const c = copy(lang);
    const map = {
      credential: { glyph: "⚠", level: "warn", text: c.credWarn },
      font: { glyph: "⚠", level: "warn", text: c.fontWarn },
      "mic-fail": { glyph: "⚠", level: "error", text: c.micFail },
      "mic-blocked": { glyph: "⚠", level: "error", text: c.micBlocked },
      unsupported: { glyph: "ⓘ", level: "info", text: c.unsupported },
    };
    const spec = map[kind] || map.unsupported;
    const node = el("div", { class: "composer-notice " + spec.level, role: spec.level === "error" ? "alert" : "status", "data-state": kind, lang });
    node.appendChild(el("span", { class: "notice__glyph", "aria-hidden": "true", text: spec.glyph }));
    node.appendChild(el("span", { text: (spec.level === "warn" ? c.warn + " · " : spec.level === "error" ? c.error + " · " : "") + spec.text }));
    return node;
  }

  root.DeskUI = {
    copy, el, memberTurn, answerCard, refusalCard, credentialDeclineCard, searching, escalateSearching,
    failureCard, rateLimitCard, offlineBanner, emptyThread, composerNotice,
  };
})(window);
