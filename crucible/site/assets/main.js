// Wires the page: one Stage, one View per [data-scene] panel, steppers, the studio terminal, copy buttons, nav state.
import { Stage, View, ENV } from './kit.js';
import * as S1 from './scenes1.js';
import * as S2 from './scenes2.js';

const SCENES = { ...S1, ...S2 };
const $ = (q, el = document) => el.querySelector(q);
const $$ = (q, el = document) => [...el.querySelectorAll(q)];

// ------------------------------------------------------------------------------------------------- studio terminal
// Real output of the CLI on the toy pack (CPU only, no model calls).
const SESSION = [
  [['$', 'crucible pack check toy --level semantic'],
    ['o', '  pass  static.hooks       credit, tasks, forge, policies, fixtures'], ['o', '  pass  fixture.cases      8 case(s) graded exactly as expected'],
    ['g', '  pass  semantic.gold      gold reaches 1.0'], ['g', '  pass  semantic.noop      no-op stays below 1.0']],
  [['$', 'crucible label --pack toy --out label'], ['k', 'labelled 24 traces -> label']],
  [['o', "  outcomes {'good': 16, 'bad': 6, 'unknown': 2}  signals 34  credited 24"]],
  [['o', '  sft 18  dpo 0 (best_vs_worst 0, correction_pivot 0)  kto 22'], ['c', '  next: crucible build toy']],
  [['$', 'crucible build toy --out build'], ['k', 'ingested 24 traces; forged 5 tasks']],
  [['g', '  accepted 5/5']],
  [['o', '  exported 5 (canonical reward.json + steps.jsonl) -> build/export']],
  [['o', "  bench {'anchor': 5, 'fresh': 2}"], ['g', '  oracle 1.0  ·  half 0.12  ·  noop 0.0'], ['c', '  report build/runs/bench-toy-build/bench/report.html']],
];
const esc = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const term = $('#term-out');
let typing = null;
function renderTerm(step) {
  if (!term) return;
  clearInterval(typing);
  const lines = SESSION.slice(0, step + 1).flat();
  const fresh = SESSION[step];
  const old = lines.slice(0, lines.length - fresh.length);
  const fmt = ([k, t]) => k === '$' ? `<span class="p">$ </span><span class="k">${esc(t)}</span>` : `<span class="${k === 'c' ? 'o' : k}">${esc(t)}</span>`;
  const head = old.map(fmt).join('\n');
  // keep the window to the most recent lines, like a real terminal
  const tail = () => { const n = term.innerHTML.split('\n'); if (n.length > 13) term.innerHTML = n.slice(-13).join('\n'); };
  if (ENV.REDUCED) { term.innerHTML = lines.map(fmt).join('\n'); tail(); return; }
  let i = 0, ch = 0; const out = [];
  term.innerHTML = head; tail();
  typing = setInterval(() => {
    const ln = fresh[i]; if (!ln) { clearInterval(typing); return; }
    if (ln[0] === '$') {
      ch = Math.min(ln[1].length, ch + 2);
      const partial = `<span class="p">$ </span><span class="k">${esc(ln[1].slice(0, ch))}</span><span class="cur"></span>`;
      term.innerHTML = [head, ...out, partial].filter(Boolean).join('\n'); tail();
      if (ch >= ln[1].length) { out.push(fmt(ln)); i++; ch = 0; }
    } else { out.push(fmt(ln)); i++; term.innerHTML = [head, ...out].filter(Boolean).join('\n'); tail(); }
  }, 28);
}
window.__studioTerm = renderTerm;
// complete at rest: the whole session is on screen before the studio figure starts replaying it
if (term) { const prev = ENV.REDUCED; ENV.REDUCED = true; renderTerm(SESSION.length - 1); ENV.REDUCED = prev; }

// ------------------------------------------------------------------------------------------------------------ stage
const stage = new Stage({ clipEl: $('header.top') });
if (!stage.failed) {
  for (const el of $$('.view[data-scene]')) {
    const make = SCENES[el.dataset.scene];
    if (!make) continue;
    const v = stage.add(new View(el, make()));
    const st = el.dataset.scene === 'studio' ? $('#studio-stepper') : el.closest('.fig')?.querySelector('.stepper');
    if (st && v.spec.steps?.length) v.bindStepper(st); else st?.remove();
  }
  window.crucible = { stage, views: Object.fromEntries(stage.views.map(v => [v.el.dataset.scene, v])) };
} else {
  $$('.stepper').forEach(s => s.remove());
  renderTerm(SESSION.length - 1);
}

// ---------------------------------------------------------------------------------------------------- copy buttons
// when the clipboard is refused, select the commands so the reader can copy them by hand
function selectText(el) { const r = document.createRange(); r.selectNodeContents(el); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); }
for (const btn of $$('.copy')) {
  btn.addEventListener('click', () => {
    const pre = btn.parentElement.querySelector('pre');
    const text = pre.innerText.split('\n').filter(l => l.trim() && !l.trim().startsWith('#')).map(l => l.replace(/^\$\s?/, '')).join('\n');
    const done = label => { btn.textContent = label; setTimeout(() => (btn.textContent = 'Copy'), 1600); };
    if (!navigator.clipboard?.writeText) { selectText(pre); done('Selected'); return; }
    navigator.clipboard.writeText(text).then(() => done('Copied'), () => { selectText(pre); done('Selected'); });
  });
}

// ------------------------------------------------------------------------------------------------------- nav state
const navLinks = $$('header .nav a[href^="#"]');
const modLinks = $$('.mod-index a[href^="#"]');
const watch = (links, rootMargin) => {
  const map = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
  const targets = [...map.keys()].map(id => document.getElementById(id)).filter(Boolean);
  const io = new IntersectionObserver(entries => {
    for (const e of entries) if (e.isIntersecting) { links.forEach(a => a.classList.remove('on')); map.get(e.target.id)?.classList.add('on'); }
  }, { rootMargin });
  targets.forEach(t => io.observe(t));
};
watch(navLinks, '-45% 0px -50% 0px');
watch(modLinks, '-35% 0px -60% 0px');
