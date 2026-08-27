// node --test tests/desk_core.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const core = require("../src/himalaya_support/static/desk-core.js");

test("E4: overview count and inbox count are one number from one store", () => {
  for (const n of [0, 1, 2, 17]) {
    const threads = Array.from({ length: n }, (_, i) => ({ id: String(i) }));
    const c = core.deskCounts({ threads, tickets: [] });
    assert.equal(c.threads, c.inbox);
    assert.equal(c.threads, n);
  }
  const c = core.deskCounts({ threads: [{}], tickets: [{ status: "open" }, { status: "resolved" }] });
  assert.equal(c.openTickets, 1);
  assert.deepEqual(core.deskCounts(undefined), { threads: 0, inbox: 0, openTickets: 0 });
});

test("E17b: chrome digits follow the interface language", () => {
  assert.equal(core.fmtNum(16, "ne"), "१६");
  assert.equal(core.fmtNum(16, "en"), "16");
});

test("E17c: relative time, 24h clock, labelled Gregorian date", () => {
  const now = "2026-08-28T10:00:00Z";
  assert.equal(core.relativeTime("2026-08-28T09:58:00Z", "ne", now), "२ मिनेट अघि");
  assert.equal(core.relativeTime("2026-08-28T09:58:00Z", "en", now), "2 min ago");
  assert.equal(core.relativeTime("2026-08-28T07:00:00Z", "ne", now), "३ घण्टा अघि");
  assert.match(core.relativeTime("2026-08-01T07:00:00Z", "ne", now), /ई\.सं\.$/);
  assert.match(core.relativeTime("2026-08-01T07:00:00Z", "en", now), / AD$/);
  assert.doesNotMatch(core.clock24("2026-08-28T03:33:00", "ne"), /AM|PM/);
});

test("E21: credential shapes", () => {
  assert.deepEqual(core.credentialShapes("my password is Sunita@2081"), ["password"]);
  assert.ok(core.credentialShapes("card 4111 1111 1111 1111 cvv 123").includes("card"));
  assert.ok(core.credentialShapes("card 4111 1111 1111 1111 cvv 123").includes("cvv"));
  assert.deepEqual(core.credentialShapes("otp 482913 aayo"), ["pin_otp"]);
  assert.deepEqual(core.credentialShapes("loan 250000 rupees"), []);
  assert.deepEqual(core.credentialShapes("30 मिनेट लक हुन्छ"), []);
});

test("E8: protected runs and caret run detection", () => {
  for (const t of ["NIMB", "SCB", "ATM", "KYC", "PIN", "OTP", "USD", "NPR", "SWIFT", "EMI", "IPO", "CIB", "QR", "SMS"]) {
    assert.ok(core.protectedReason(t), t);
  }
  assert.equal(core.protectedReason("khata"), null);
  assert.equal(core.protectedReason("a1"), "digits");
  assert.deepEqual(core.runAtCaret("mero khata", 10), { text: "khata", start: 5, end: 10 });
  assert.equal(core.runAtCaret("mero khata", 7), null);
  assert.equal(core.runAtCaret("test@example", 12), null);
});

test("E10: language of the question after protected tokens", () => {
  assert.equal(core.detectLanguage("मेरो KYC अपडेट"), "ne");
  assert.equal(core.detectLanguage("How do I update my KYC?"), "en");
});
