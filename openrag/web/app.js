/* open-rag web UI — zero build step, plain ES module. */

const API = '/v1';
const SUPPORTED = '.pdf,.docx,.xlsx,.csv,.md,.markdown,.txt,.html,.htm';

const state = {
  route: 'documents',
  health: null,
  kbs: [],
  kbId: null,
  docs: [],
  docTimer: null,
  datasets: [],
  datasetId: null,
  cases: [],
  runs: [],
  run: null,
  runTimer: null,
};

/* ------------------------------------------------------------------ utils */
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[char]);
}

function fmtBytes(bytes) {
  const size = Number(bytes) || 0;
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function fmtTime(iso) {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  });
}

const fmtPct = (value) => (typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—');
const fmtNum = (value, digits = 3) => (typeof value === 'number' ? value.toFixed(digits) : '—');

const STATUS = {
  pending: ['待处理', 'warn'],
  processing: ['解析中', 'info'],
  ready: ['已就绪', 'ok'],
  failed: ['失败', 'err'],
  running: ['运行中', 'info'],
  done: ['已完成', 'ok'],
};

function statusPill(status) {
  const [label, tone] = STATUS[status] || [status || '未知', ''];
  return `<span class="pill ${tone}">${escapeHtml(label)}</span>`;
}

async function api(path, { method = 'GET', json, form } = {}) {
  const init = { method, headers: {} };
  if (json !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(json);
  }
  if (form) init.body = form;

  const response = await fetch(API + path, init);
  if (!response.ok) {
    let detail = `请求失败（HTTP ${response.status}）`;
    try {
      const data = await response.json();
      if (typeof data?.detail === 'string') detail = data.detail;
      else if (Array.isArray(data?.detail)) detail = data.detail.map((item) => item.msg).join('；');
    } catch { /* keep the default message */ }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

function toast(message, tone = '') {
  const root = $('#toast-root');
  const node = document.createElement('div');
  node.className = `toast ${tone}`;
  node.textContent = message;
  root.append(node);
  setTimeout(() => node.remove(), tone === 'err' ? 6000 : 4000);
}

function openModal({ title, body, footer = '', onMount, onClose }) {
  const root = $('#modal-root');
  root.innerHTML = `
    <div class="modal-backdrop">
      <div class="modal" role="dialog" aria-modal="true" aria-label="${escapeHtml(title)}">
        <div class="modal-head">
          <h3>${escapeHtml(title)}</h3>
          <button class="icon-btn" data-close aria-label="关闭">✕</button>
        </div>
        <div class="modal-body">${body}</div>
        ${footer ? `<div class="modal-foot">${footer}</div>` : ''}
      </div>
    </div>`;
  let closed = false;
  const onKey = (event) => { if (event.key === 'Escape') close(); };
  const close = () => {
    if (closed) return;
    closed = true;
    root.innerHTML = '';
    document.removeEventListener('keydown', onKey);
    onClose?.();
  };
  document.addEventListener('keydown', onKey);
  $('[data-close]', root).addEventListener('click', close);
  $('.modal-backdrop', root).addEventListener('mousedown', (event) => {
    if (event.target === event.currentTarget) close();
  });
  onMount?.(root, close);
  return { root, close };
}

function confirmDialog(title, message, confirmText = '确认删除') {
  return new Promise((resolve) => {
    let answered = false;
    openModal({
      title,
      body: `<p class="muted" style="font-size:13px">${escapeHtml(message)}</p>`,
      footer: `<button class="btn" data-cancel>取消</button><button class="btn danger" data-ok>${escapeHtml(confirmText)}</button>`,
      onClose: () => { if (!answered) resolve(false); },
      onMount: (root, close) => {
        $('[data-cancel]', root).addEventListener('click', close);
        $('[data-ok]', root).addEventListener('click', () => {
          answered = true;
          close();
          resolve(true);
        });
      },
    });
  });
}

function emptyState({ icon = '◈', title, text, action = '' }) {
  return `<div class="empty">
    <div class="empty-icon">${icon}</div>
    <h2>${escapeHtml(title)}</h2>
    <p>${escapeHtml(text)}</p>
    ${action}
  </div>`;
}

/* ---------------------------------------------------------------- health */
async function loadHealth() {
  try {
    const health = await api('/health');
    state.health = health;
    const provider = health.embedding?.provider ?? 'unknown';
    const model = health.embedding?.model ?? '';
    const isHash = provider === 'hash';
    const badge = $('#embed-badge');
    badge.className = `badge ${isHash ? 'warn' : 'ghost'}`;
    badge.innerHTML = `<span class="dot"></span>${escapeHtml(isHash ? '内置 hash 向量' : model)}`;
    badge.title = isHash
      ? '当前使用内置哈希向量（离线兜底，无需下载模型）。生产环境建议设置 OPENRAG_EMBEDDING_PROVIDER=sentence-transformers'
      : `embedding: ${provider} / ${model}`;
  } catch { /* the header badge is decorative */ }
}

/* ======================================================================== */
/* documents page                                                          */
/* ======================================================================== */
async function renderDocumentsPage() {
  const view = $('#view');
  view.innerHTML = `
    <div class="layout">
      <aside class="card sidebar">
        <div class="sidebar-head">
          <h2>知识库</h2>
          <button class="icon-btn" id="new-kb" title="新建知识库">+</button>
        </div>
        <ul class="item-list" id="kb-list"></ul>
        <div class="sidebar-foot">共 <span id="kb-count">0</span> 个知识库</div>
      </aside>
      <section class="content" id="kb-view"></section>
    </div>`;

  $('#new-kb').addEventListener('click', openKbModal);
  $('#kb-list').addEventListener('click', (event) => {
    const button = event.target.closest('[data-kb]');
    if (!button || button.dataset.kb === state.kbId) return;
    state.kbId = button.dataset.kb;
    renderKbList();
    renderKbView();
  });

  try {
    state.kbs = await api('/knowledge-bases');
  } catch (error) {
    $('#kb-view').innerHTML = emptyState({
      icon: '!',
      title: '无法连接后端',
      text: `${error.message}。请确认服务已启动：python -m openrag`,
    });
    return;
  }

  if (!state.kbs.some((kb) => kb.id === state.kbId)) {
    state.kbId = state.kbs[0]?.id ?? null;
  }
  renderKbList();
  renderKbView();
}

function renderKbList() {
  const list = $('#kb-list');
  if (!state.kbs.length) {
    list.innerHTML = '<li style="padding:18px 12px" class="faint">还没有知识库</li>';
  } else {
    list.innerHTML = state.kbs
      .map((kb) => `
        <li>
          <button class="item ${kb.id === state.kbId ? 'active' : ''}" data-kb="${kb.id}">
            <span class="item-icon">▤</span>
            <span class="item-main">
              <span class="item-name">${escapeHtml(kb.name)}</span>
              <span class="item-meta">${kb.document_count} 文档 · ${kb.chunk_count} 分块</span>
            </span>
          </button>
        </li>`)
      .join('');
  }
  $('#kb-count').textContent = state.kbs.length;
}

function currentKb() {
  return state.kbs.find((kb) => kb.id === state.kbId) || null;
}

function renderKbView() {
  const container = $('#kb-view');
  const kb = currentKb();

  if (!kb) {
    container.innerHTML = `<div class="card">${emptyState({
      icon: '▤',
      title: '还没有知识库',
      text: '创建一个知识库，上传文档后即可通过 /v1/retrieve 检索。',
      action: '<button class="btn primary" id="empty-create">新建知识库</button>',
    })}</div>`;
    $('#empty-create').addEventListener('click', openKbModal);
    return;
  }

  const maxMb = state.health?.max_upload_mb ?? 50;
  container.innerHTML = `
    <div class="page-head">
      <div>
        <h1>${escapeHtml(kb.name)}</h1>
        <p class="sub"><span id="kb-meta">${escapeHtml(kb.description || '暂无描述')} · ${kb.document_count} 文档 · ${kb.chunk_count} 分块</span></p>
      </div>
      <div class="actions">
        <button class="btn danger" id="delete-kb">删除知识库</button>
      </div>
    </div>

    <label class="dropzone" id="dropzone" role="button" tabindex="0" aria-label="上传文档">
      <input type="file" id="file-input" multiple hidden accept="${SUPPORTED}" />
      <div class="dz-icon">⇪</div>
      <div class="dz-title">拖拽文件到此处，或点击选择</div>
      <div class="dz-hint">支持 PDF / DOCX / XLSX / CSV / MD / TXT / HTML · 单文件 ≤ ${maxMb}MB</div>
    </label>

    <div class="card section">
      <div class="section-head">
        <h3>文档</h3>
        <span class="muted" id="docs-hint">加载中…</span>
      </div>
      <div class="table-wrap">
        <table class="table">
          <thead>
            <tr><th>文件名</th><th>状态</th><th>分块</th><th>大小</th><th>更新时间</th><th></th></tr>
          </thead>
          <tbody id="docs-body">
            <tr><td colspan="6" class="muted" style="padding:18px">加载中…</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <details class="card section search-panel">
      <summary>检索测试</summary>
      <div class="search-body">
        <form class="inline-form" id="search-form">
          <div class="grow field" style="margin:0">
            <input type="search" id="search-query" placeholder="输入问题，例如：E-201 设备的最大工作压力是多少？" autocomplete="off" />
          </div>
          <select id="search-mode" title="检索模式">
            <option value="hybrid">混合检索</option>
            <option value="vector">向量检索</option>
            <option value="keyword">关键词检索</option>
          </select>
          <select id="search-topk" title="返回条数">
            <option value="3">Top 3</option>
            <option value="5" selected>Top 5</option>
            <option value="10">Top 10</option>
          </select>
          <button class="btn primary" type="submit">检索</button>
        </form>
        <div id="search-results" class="search-body" style="gap:8px"></div>
      </div>
    </details>`;

  const dropzone = $('#dropzone');
  const fileInput = $('#file-input');
  fileInput.addEventListener('change', () => {
    uploadFiles(fileInput.files);
    fileInput.value = '';
  });
  ['dragenter', 'dragover'].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add('dragover');
    }),
  );
  ['dragleave', 'drop'].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove('dragover');
    }),
  );
  dropzone.addEventListener('drop', (event) => uploadFiles(event.dataTransfer.files));
  dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      fileInput.click();
    }
  });

  $('#delete-kb').addEventListener('click', async () => {
    const ok = await confirmDialog(
      '删除知识库',
      `确定删除「${kb.name}」吗？其中的文档、分块和评测集都会被删除，且无法恢复。`,
    );
    if (!ok) return;
    try {
      await api(`/knowledge-bases/${kb.id}`, { method: 'DELETE' });
      state.kbs = state.kbs.filter((item) => item.id !== kb.id);
      state.kbId = state.kbs[0]?.id ?? null;
      state.datasets = state.datasets.filter((item) => item.kb_id !== kb.id);
      renderKbList();
      renderKbView();
      toast('知识库已删除', 'ok');
    } catch (error) {
      toast(error.message, 'err');
    }
  });

  $('#docs-body').addEventListener('click', async (event) => {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    const doc = state.docs.find((item) => item.id === button.dataset.id);
    if (!doc) return;

    if (button.dataset.action === 'chunks') {
      showChunks(doc, 0);
      return;
    }
    if (button.dataset.action === 'delete') {
      const ok = await confirmDialog('删除文档', `确定删除「${doc.name}」及其所有分块吗？`);
      if (!ok) return;
      try {
        await api(`/documents/${doc.id}`, { method: 'DELETE' });
        await loadDocs();
        await refreshKbMeta();
        toast('文档已删除', 'ok');
      } catch (error) {
        toast(error.message, 'err');
      }
    }
  });

  $('#search-form').addEventListener('submit', runSearch);
  loadDocs();
}

async function loadDocs() {
  if (!state.kbId) return;
  const kbId = state.kbId;
  try {
    const docs = await api(`/knowledge-bases/${kbId}/documents`);
    if (kbId !== state.kbId) return;
    state.docs = docs;
    renderDocs();
    ensureDocPolling();
  } catch (error) {
    toast(error.message, 'err');
  }
}

function renderDocs() {
  const body = $('#docs-body');
  if (!body) return;

  if (!state.docs.length) {
    body.innerHTML = `<tr><td colspan="6" class="muted" style="padding:22px;text-align:center">
      还没有文档，先上传一个试试
    </td></tr>`;
    const hint = $('#docs-hint');
    if (hint) hint.textContent = '0 个文档';
    return;
  }

  body.innerHTML = state.docs.map((doc) => {
    const ext = (doc.ext || '').replace('.', '').toUpperCase() || 'FILE';
    return `<tr>
      <td>
        <div class="doc-name">
          <span class="file-ext">${escapeHtml(ext)}</span>
          <div style="min-width:0">
            <div class="name">${escapeHtml(doc.name)}</div>
            ${doc.error ? `<div class="err-text">${escapeHtml(doc.error)}</div>` : ''}
          </div>
        </div>
      </td>
      <td>${statusPill(doc.status)}</td>
      <td class="num">${doc.chunk_count}</td>
      <td class="num">${fmtBytes(doc.size)}</td>
      <td class="muted">${fmtTime(doc.updated_at)}</td>
      <td class="row-actions">
        <button class="link-btn" data-action="chunks" data-id="${doc.id}" ${doc.status === 'ready' ? '' : 'disabled'}>分块</button>
        <button class="link-btn danger" data-action="delete" data-id="${doc.id}">删除</button>
      </td>
    </tr>`;
  }).join('');

  const hint = $('#docs-hint');
  if (hint) {
    const ready = state.docs.filter((doc) => doc.status === 'ready').length;
    const busy = state.docs.filter((doc) => ['pending', 'processing'].includes(doc.status)).length;
    hint.textContent = `${state.docs.length} 个文档${busy ? ` · ${busy} 个处理中` : ` · ${ready} 个已就绪`}`;
  }
}

function ensureDocPolling() {
  const busy = state.docs.some((doc) => ['pending', 'processing'].includes(doc.status));
  if (busy && !state.docTimer) {
    state.docTimer = setInterval(async () => {
      if (state.route !== 'documents' || !state.kbId) {
        clearDocTimer();
        return;
      }
      await loadDocs();
      if (!state.docs.some((doc) => ['pending', 'processing'].includes(doc.status))) {
        clearDocTimer();
        refreshKbMeta();
      }
    }, 1500);
  }
}

function clearDocTimer() {
  if (state.docTimer) {
    clearInterval(state.docTimer);
    state.docTimer = null;
  }
}

async function refreshKbMeta() {
  try {
    state.kbs = await api('/knowledge-bases');
    renderKbList();
    const kb = currentKb();
    const meta = $('#kb-meta');
    if (kb && meta) {
      meta.textContent = `${kb.description || '暂无描述'} · ${kb.document_count} 文档 · ${kb.chunk_count} 分块`;
    }
  } catch { /* non critical */ }
}

async function uploadFiles(fileList) {
  const files = [...(fileList || [])];
  if (!files.length || !state.kbId) return;

  const form = new FormData();
  files.forEach((file) => form.append('files', file));

  try {
    const result = await api(`/knowledge-bases/${state.kbId}/documents`, { method: 'POST', form });
    if (result.rejected?.length) {
      const reasons = result.rejected.map((item) => `${item.name}（${item.reason}）`).join('；');
      toast(`已跳过 ${result.rejected.length} 个文件：${reasons}`, 'warn');
    }
    if (result.accepted?.length) {
      toast(`已接收 ${result.accepted.length} 个文件，正在后台解析`, 'ok');
    }
    await loadDocs();
    ensureDocPolling();
    refreshKbMeta();
  } catch (error) {
    toast(error.message, 'err');
  }
}

async function showChunks(doc, offset) {
  let page;
  try {
    page = await api(`/documents/${doc.id}/chunks?offset=${offset}&limit=20`);
  } catch (error) {
    toast(error.message, 'err');
    return;
  }

  const items = page.items.map((item) => `
    <div class="hit">
      <div class="hit-head">
        <span class="hit-rank">#${item.ordinal + 1}</span>
        ${item.page ? `<span class="pill">第 ${item.page} 页</span>` : ''}
        <span class="hit-score">${item.tokens || item.text.length} tokens</span>
        <button class="link-btn" data-copy="${item.id}" title="复制分块 ID，用于评测标注">复制 ID</button>
      </div>
      <div class="hit-text">${escapeHtml(item.text)}</div>
    </div>`).join('');

  const footer = page.total > 20
    ? `<button class="btn" data-page="prev" ${offset === 0 ? 'disabled' : ''}>上一页</button>
       <button class="btn" data-page="next" ${offset + 20 >= page.total ? 'disabled' : ''}>下一页</button>`
    : '';

  const { close } = openModal({
    title: `分块预览 · ${doc.name}`,
    body: `
      <div class="row" style="justify-content:space-between;margin-bottom:10px">
        <span class="muted">共 ${page.total} 个分块</span>
        <span class="mono faint">${page.total ? offset + 1 : 0}–${Math.min(offset + 20, page.total)} / ${page.total}</span>
      </div>
      <div class="search-body" style="gap:8px">${items || '<p class="faint">没有分块</p>'}</div>`,
    footer,
  });

  $$('[data-page]').forEach((button) => {
    button.addEventListener('click', () => {
      close();
      showChunks(doc, button.dataset.page === 'next' ? offset + 20 : Math.max(0, offset - 20));
    });
  });
  $$('[data-copy]').forEach((button) => {
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        toast('分块 ID 已复制', 'ok');
      } catch {
        toast('复制失败，请手动选中文本', 'warn');
      }
    });
  });
}

async function runSearch(event) {
  event.preventDefault();
  const query = $('#search-query').value.trim();
  if (!query) return;
  const box = $('#search-results');
  box.innerHTML = '<p class="faint">检索中…</p>';

  try {
    const result = await api('/retrieve', {
      method: 'POST',
      json: {
        kb_id: state.kbId,
        query,
        top_k: Number($('#search-topk').value),
        mode: $('#search-mode').value,
      },
    });

    if (!result.hits.length) {
      box.innerHTML = '<p class="faint">没有检索到相关内容，试试换个说法或先上传文档。</p>';
      return;
    }

    box.innerHTML = `
      <div class="row faint" style="font-size:12px">
        <span>召回 ${result.recalled ?? result.hits.length} 条</span>
        ${result.dropped ? `<span>·</span><span>过滤 ${result.dropped} 条</span>` : ''}
        <span>·</span>
        <span>返回 ${result.returned ?? result.hits.length} 条</span>
        <span>·</span>
        <span>知识库共 ${result.total_chunks} 个分块</span>
        <span>·</span>
        <span>${result.latency_ms.toFixed(1)} ms</span>
        ${result.rerank_applied ? '<span>·</span><span>已重排</span>' : ''}
      </div>
      ${result.rerank_warning ? `<div class="notice">rerank 未生效，已回退到混合检索：${escapeHtml(result.rerank_warning)}</div>` : ''}
      ${result.hits.map((hit, index) => `
        <div class="hit">
          <div class="hit-head">
            <span class="hit-rank">#${index + 1}</span>
            <span class="hit-doc">${escapeHtml(hit.doc_name)}</span>
            ${hit.page ? `<span class="pill">第 ${hit.page} 页</span>` : ''}
            ${hit.tokens ? `<span class="pill">${hit.tokens} tokens</span>` : ''}
            <span class="hit-score">${Number(typeof hit.scores?.rerank === 'number' ? hit.scores.rerank : hit.score).toFixed(3)}</span>
          </div>
          <div class="hit-text">${escapeHtml(hit.text)}</div>
          ${hit.context ? `
            <details class="hit-context">
              <summary>父块上下文</summary>
              <div class="hit-text">${escapeHtml(hit.context)}</div>
            </details>` : ''}
          <div class="hit-scores">
            vector ${hit.scores.vector.toFixed(3)} · keyword ${hit.scores.keyword.toFixed(2)}
            ${typeof hit.scores.rerank === 'number' ? ` · rerank ${hit.scores.rerank.toFixed(3)}` : ''}
            · ${result.filter_score || 'fused'} 过滤
          </div>
        </div>`).join('')}`;
  } catch (error) {
    box.innerHTML = '';
    toast(error.message, 'err');
  }
}

function openKbModal() {
  openModal({
    title: '新建知识库',
    body: `
      <div class="field">
        <label for="kb-name">名称</label>
        <input id="kb-name" type="text" placeholder="例如：设备手册" maxlength="80" />
      </div>
      <div class="field">
        <label for="kb-desc">描述（可选）</label>
        <input id="kb-desc" type="text" placeholder="这个知识库放什么内容" maxlength="500" />
      </div>`,
    footer: '<button class="btn" data-cancel>取消</button><button class="btn primary" data-save>创建</button>',
    onMount: (root, close) => {
      const nameInput = $('#kb-name', root);
      nameInput.focus();

      const save = async () => {
        const name = nameInput.value.trim();
        if (!name) {
          toast('请输入知识库名称', 'warn');
          return;
        }
        try {
          const kb = await api('/knowledge-bases', {
            method: 'POST',
            json: { name, description: $('#kb-desc', root).value.trim() },
          });
          close();
          state.kbId = kb.id;
          await renderDocumentsPage();
          toast('知识库已创建', 'ok');
        } catch (error) {
          toast(error.message, 'err');
        }
      };

      $('[data-save]', root).addEventListener('click', save);
      $('[data-cancel]', root).addEventListener('click', close);
      nameInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') save();
      });
    },
  });
}

/* ======================================================================== */
/* evaluation page                                                         */
/* ======================================================================== */
async function renderEvaluationPage() {
  const view = $('#view');
  view.innerHTML = `
    <div class="layout">
      <aside class="card sidebar">
        <div class="sidebar-head">
          <h2>评测集</h2>
          <button class="icon-btn" id="new-dataset" title="新建评测集">+</button>
        </div>
        <ul class="item-list" id="dataset-list"></ul>
        <div class="sidebar-foot">共 <span id="dataset-count">0</span> 个评测集</div>
      </aside>
      <section class="content" id="dataset-view"></section>
    </div>`;

  $('#new-dataset').addEventListener('click', openDatasetModal);
  $('#dataset-list').addEventListener('click', (event) => {
    const button = event.target.closest('[data-dataset]');
    if (!button || button.dataset.dataset === state.datasetId) return;
    selectDataset(button.dataset.dataset);
  });

  try {
    const [kbs, datasets] = await Promise.all([api('/knowledge-bases'), api('/eval/datasets')]);
    state.kbs = kbs;
    state.datasets = datasets;
  } catch (error) {
    $('#dataset-view').innerHTML = emptyState({
      icon: '!',
      title: '无法加载评测数据',
      text: error.message,
    });
    return;
  }

  if (!state.datasets.some((item) => item.id === state.datasetId)) {
    state.datasetId = state.datasets[0]?.id ?? null;
  }
  renderDatasetList();
  await renderDatasetView();
}

function renderDatasetList() {
  const list = $('#dataset-list');
  if (!state.datasets.length) {
    list.innerHTML = '<li style="padding:18px 12px" class="faint">还没有评测集</li>';
  } else {
    list.innerHTML = state.datasets.map((dataset) => `
      <li>
        <button class="item ${dataset.id === state.datasetId ? 'active' : ''}" data-dataset="${dataset.id}">
          <span class="item-icon">◎</span>
          <span class="item-main">
            <span class="item-name">${escapeHtml(dataset.name)}</span>
            <span class="item-meta">${dataset.case_count} 条用例</span>
          </span>
        </button>
      </li>`).join('');
  }
  $('#dataset-count').textContent = state.datasets.length;
}

function currentDataset() {
  return state.datasets.find((item) => item.id === state.datasetId) || null;
}

async function selectDataset(id) {
  state.datasetId = id;
  state.run = null;
  renderDatasetList();
  await renderDatasetView();
}

async function renderDatasetView() {
  const container = $('#dataset-view');
  const dataset = currentDataset();

  if (!dataset) {
    container.innerHTML = `<div class="card">${emptyState({
      icon: '◎',
      title: '还没有评测集',
      text: '评测集用来回答一个问题：你的检索，在真实问题上到底准不准？',
      action: '<button class="btn primary" id="empty-dataset">新建评测集</button>',
    })}</div>`;
    const button = $('#empty-dataset');
    if (button) button.addEventListener('click', openDatasetModal);
    return;
  }

  let cases;
  let runs;
  try {
    [cases, runs] = await Promise.all([
      api(`/eval/datasets/${dataset.id}/cases`),
      api(`/eval/runs?dataset_id=${dataset.id}`),
    ]);
  } catch (error) {
    container.innerHTML = `<div class="card">${emptyState({ icon: '!', title: '加载失败', text: error.message })}</div>`;
    return;
  }

  state.cases = cases;
  state.runs = runs;

  const kb = state.kbs.find((item) => item.id === dataset.kb_id);
  container.innerHTML = `
    <div class="page-head">
      <div>
        <h1>${escapeHtml(dataset.name)}</h1>
        <p class="sub">${escapeHtml(dataset.description || '暂无描述')} · 知识库 ${escapeHtml(kb?.name ?? '已删除')} · ${cases.length} 条用例</p>
      </div>
      <div class="actions">
        <button class="btn danger" id="delete-dataset">删除评测集</button>
      </div>
    </div>

    <div class="card section">
      <div class="section-head">
        <h3>用例</h3>
        <button class="btn sm" id="add-case">+ 添加用例</button>
      </div>
      <div class="table-wrap">
        <table class="table">
          <thead><tr><th>问题</th><th>标注</th><th></th></tr></thead>
          <tbody id="case-body"></tbody>
        </table>
      </div>
    </div>

    <div class="card section">
      <div class="section-head">
        <h3>运行评测</h3>
        <span class="muted">对当前知识库执行检索，并计算 Hit / Recall / MRR / nDCG</span>
      </div>
      <form class="inline-form" id="run-form">
        <select id="run-topk" title="Top K">
          <option value="3">Top 3</option>
          <option value="5" selected>Top 5</option>
          <option value="10">Top 10</option>
        </select>
        <select id="run-mode" title="检索模式">
          <option value="hybrid">混合检索</option>
          <option value="vector">向量检索</option>
          <option value="keyword">关键词检索</option>
        </select>
        <button class="btn primary" type="submit" id="run-btn" ${cases.length ? '' : 'disabled'}>开始评测</button>
        <span class="faint" id="run-hint" style="font-size:12px;${cases.length ? 'display:none' : ''}">先添加至少一条用例</span>
      </form>
      <div class="divider" style="margin:16px 0"></div>
      <div class="run-list" id="run-list"></div>
    </div>

    <div id="run-detail"></div>`;

  renderCaseTable();
  renderRuns();
  renderRunDetail(state.run);

  $('#add-case').addEventListener('click', openCaseModal);
  $('#case-body').addEventListener('click', async (event) => {
    const button = event.target.closest('button[data-case]');
    if (!button) return;
    try {
      await api(`/eval/cases/${button.dataset.case}`, { method: 'DELETE' });
      state.cases = state.cases.filter((item) => item.id !== button.dataset.case);
      renderCaseTable();
      toggleRunButton();
      refreshDatasetMeta();
      toast('用例已删除', 'ok');
    } catch (error) {
      toast(error.message, 'err');
    }
  });

  $('#run-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    await startRun();
  });

  $('#run-list').addEventListener('click', (event) => {
    const button = event.target.closest('[data-run]');
    if (button) loadRun(button.dataset.run);
  });

  $('#delete-dataset').addEventListener('click', async () => {
    const ok = await confirmDialog('删除评测集', `确定删除「${dataset.name}」及其全部用例和评测记录吗？`);
    if (!ok) return;
    try {
      await api(`/eval/datasets/${dataset.id}`, { method: 'DELETE' });
      state.datasets = state.datasets.filter((item) => item.id !== dataset.id);
      state.datasetId = state.datasets[0]?.id ?? null;
      state.run = null;
      renderDatasetList();
      await renderDatasetView();
      toast('评测集已删除', 'ok');
    } catch (error) {
      toast(error.message, 'err');
    }
  });

  if (runs.length) {
    loadRun(runs[0].id);
  }
}

function renderCaseTable() {
  const body = $('#case-body');
  if (!body) return;

  if (!state.cases.length) {
    body.innerHTML = `<tr><td colspan="3" class="muted" style="padding:22px;text-align:center">
      还没有用例。一条好的用例 = 一个真实问题 + 答案所在的文档或关键词。
    </td></tr>`;
    return;
  }

  body.innerHTML = state.cases.map((item) => {
    const marks = [];
    const hasChunks = (item.reference_chunk_ids || []).length > 0;
    if (hasChunks) marks.push(`<span class="pill ok">${item.reference_chunk_ids.length} 个参考分块</span>`);
    if (item.reference_doc_ids.length) {
      marks.push(`<span class="pill ${hasChunks ? '' : 'warn'}">${item.reference_doc_ids.length} 个参考文档</span>`);
    }
    if (item.keywords.length) marks.push(`<span class="pill" title="${escapeHtml(item.keywords.join('、'))}">${item.keywords.length} 个关键词</span>`);
    return `<tr>
      <td style="max-width:520px">${escapeHtml(item.question)}</td>
      <td><div class="row" style="gap:6px">${marks.join('') || '<span class="faint">—</span>'}</div></td>
      <td class="row-actions">
        <button class="link-btn danger" data-case="${item.id}">删除</button>
      </td>
    </tr>`;
  }).join('');
}

function toggleRunButton() {
  const button = $('#run-btn');
  const hint = $('#run-hint');
  if (button) button.disabled = state.cases.length === 0;
  if (hint) hint.style.display = state.cases.length ? 'none' : '';
}

function renderRuns() {
  const list = $('#run-list');
  if (!list) return;

  if (!state.runs.length) {
    list.innerHTML = '<p class="faint" style="font-size:12.5px">还没有评测记录</p>';
    return;
  }

  list.innerHTML = state.runs.map((run) => {
    const metrics = run.metrics || {};
    const score = typeof metrics.hit_rate === 'number' ? `Hit ${fmtPct(metrics.hit_rate)}` : '';
    return `<button class="run-item ${state.run?.id === run.id ? 'active' : ''}" data-run="${run.id}">
      <span class="run-main">
        <span class="run-title">${fmtTime(run.created_at)} · ${escapeHtml(run.config?.mode ?? 'hybrid')} · Top ${run.config?.top_k ?? 5}</span>
        <span class="run-sub">${run.case_count ?? 0} 条用例</span>
      </span>
      ${score ? `<span class="run-score">${score}</span>` : ''}
      ${statusPill(run.status)}
    </button>`;
  }).join('');
}

async function loadRun(runId) {
  try {
    const run = await api(`/eval/runs/${runId}`);
    state.run = run;
    renderRuns();
    renderRunDetail(run);
  } catch (error) {
    toast(error.message, 'err');
  }
}

function metricCard(label, value, sub = '') {
  return `<div class="metric">
    <div class="metric-label">${escapeHtml(label)}</div>
    <div class="metric-value">${escapeHtml(value)}</div>
    ${sub ? `<div class="metric-sub">${escapeHtml(sub)}</div>` : ''}
  </div>`;
}

function renderRunDetail(run) {
  const container = $('#run-detail');
  if (!container) return;

  if (!run) {
    container.innerHTML = `<div class="card section"><p class="faint" style="font-size:12.5px">运行一次评测，或在上方选择历史记录查看详情。</p></div>`;
    return;
  }

  if (run.status === 'pending' || run.status === 'running') {
    container.innerHTML = `<div class="card section">
      <div class="row">${statusPill(run.status)}<span class="muted">正在对 ${run.case_count || '…'} 条用例执行检索</span></div>
      <div class="skeleton" style="margin-top:14px;width:70%"></div>
      <div class="skeleton" style="margin-top:8px;width:40%"></div>
    </div>`;
    return;
  }

  if (run.status === 'failed') {
    container.innerHTML = `<div class="card section">
      <div class="row">${statusPill('failed')}<span class="muted">评测失败</span></div>
      <p class="err-text" style="margin-top:8px;font-size:12.5px">${escapeHtml(run.error || '未知错误')}</p>
    </div>`;
    return;
  }

  const metrics = run.metrics || {};
  const k = Number(metrics.k) || Number(run.config?.top_k) || 5;
  const results = run.results || [];

  const labels = [];
  if (metrics.chunk_labeled) labels.push(`分块级 ${metrics.chunk_labeled}`);
  if (metrics.doc_labeled) labels.push(`文档级 ${metrics.doc_labeled}`);
  if (metrics.keyword_labeled) labels.push(`关键词 ${metrics.keyword_labeled}`);
  const labelNote = labels.length ? ` · 标注：${labels.join(' / ')}` : '';
  const labelWarning =
    !metrics.chunk_labeled && metrics.doc_labeled
      ? `<div class="notice">当前用例只有文档级标注：同一文档里的所有分块都会被算作相关，指标偏乐观。建议补充「参考分块」或「关键词」。</div>`
      : '';

  container.innerHTML = `
    <div class="card section">
      <div class="section-head">
        <h3>评测结果</h3>
        <span class="muted">${escapeHtml(run.config?.mode ?? 'hybrid')} · Top ${k} · ${metrics.cases ?? 0} 条有效用例${labelNote}</span>
      </div>
      <div class="metric-grid">
        ${metricCard(`Hit@${k}`, fmtPct(metrics.hit_rate), '至少命中一个相关分块')}
        ${metricCard(`Recall@${k}`, fmtPct(metrics.recall), '相关文档/分块/关键词被召回比例')}
        ${metricCard('MRR', fmtNum(metrics.mrr), '首个相关结果的平均排名')}
        ${metricCard(`nDCG@${k}`, fmtNum(metrics.ndcg), '排序质量')}
        ${metricCard('P95 延迟', `${Math.round(metrics.latency_p95 ?? 0)} ms`, `P50 ${Math.round(metrics.latency_p50 ?? 0)} ms`)}
      </div>
      ${labelWarning}
    </div>

    <div class="card section">
      <div class="section-head">
        <h3>逐条结果</h3>
        <span class="muted">点击用例展开检索到的分块</span>
      </div>
      <div class="search-body" style="gap:8px">
        ${results.map((item) => caseResult(item, k)).join('') || '<p class="faint">没有结果</p>'}
      </div>
    </div>`;
}

function caseResult(item, k) {
  const metrics = item.metrics || {};
  const hit = (metrics.hit ?? 0) >= 1;
  const hits = item.hits || [];

  return `<details class="case-result">
    <summary>
      <span class="case-q">${escapeHtml(item.question)}</span>
      <span class="case-metrics">
        <span class="mini-metric ${hit ? 'good' : 'bad'}">Hit ${hit ? '1' : '0'}</span>
        <span class="mini-metric">R ${fmtPct(metrics.recall)}</span>
        <span class="mini-metric">MRR ${fmtNum(metrics.mrr, 2)}</span>
        <span class="mini-metric">nDCG ${fmtNum(metrics.ndcg, 2)}</span>
        <span class="mini-metric">${Math.round(item.latency_ms)} ms</span>
      </span>
    </summary>
    <div class="case-hits">
      ${hits.map((hit, index) => `
        <div class="case-hit ${hit.relevant ? 'relevant' : ''}">
          <div class="meta">
            <span>#${index + 1}</span>
            <span>${escapeHtml(hit.doc_name)}</span>
            ${hit.page ? `<span>第 ${hit.page} 页</span>` : ''}
            <span>chunk ${hit.ordinal}</span>
            ${hit.tokens ? `<span>${hit.tokens} tokens</span>` : ''}
            <span>score ${Number(hit.score).toFixed(4)}</span>
            <span>${hit.relevant ? '相关' : '不相关'}</span>
          </div>
          <div>${escapeHtml(hit.preview)}</div>
        </div>`).join('') || '<p class="faint">没有检索结果</p>'}
    </div>
  </details>`;
}

async function startRun() {
  if (!state.datasetId) return;
  const button = $('#run-btn');
  button.disabled = true;

  try {
    const run = await api(`/eval/datasets/${state.datasetId}/runs`, {
      method: 'POST',
      json: { top_k: Number($('#run-topk').value), mode: $('#run-mode').value },
    });
    state.runs.unshift(run);
    state.run = run;
    renderRuns();
    renderRunDetail(run);
    pollRun(run.id);
  } catch (error) {
    toast(error.message, 'err');
  } finally {
    button.disabled = false;
  }
}

function pollRun(runId) {
  clearTimeout(state.runTimer);
  state.runTimer = setTimeout(async () => {
    if (state.route !== 'evaluate') return;
    try {
      const run = await api(`/eval/runs/${runId}`);
      const index = state.runs.findIndex((item) => item.id === runId);
      if (index >= 0) state.runs[index] = { ...state.runs[index], ...run };
      const isCurrent = state.run?.id === runId;
      if (isCurrent) {
        state.run = run;
        renderRunDetail(run);
      }
      renderRuns();
      if (run.status === 'pending' || run.status === 'running') {
        pollRun(runId);
      } else {
        refreshDatasetMeta();
      }
    } catch (error) {
      toast(error.message, 'err');
    }
  }, 800);
}

async function refreshDatasetMeta() {
  try {
    const datasets = await api('/eval/datasets');
    state.datasets = datasets;
    renderDatasetList();
    const dataset = currentDataset();
    if (dataset) {
      const sub = $('#dataset-view .sub');
      if (sub) {
        const kb = state.kbs.find((item) => item.id === dataset.kb_id);
        sub.textContent = `${dataset.description || '暂无描述'} · 知识库 ${kb?.name ?? '已删除'} · ${dataset.case_count} 条用例`;
      }
    }
  } catch { /* non critical */ }
}

function openDatasetModal() {
  if (!state.kbs.length) {
    toast('请先在「知识库」页面创建一个知识库', 'warn');
    return;
  }

  openModal({
    title: '新建评测集',
    body: `
      <div class="field">
        <label for="ds-name">名称</label>
        <input id="ds-name" type="text" placeholder="例如：设备手册 · 运维问答集" maxlength="80" />
      </div>
      <div class="field">
        <label for="ds-kb">关联知识库</label>
        <select id="ds-kb">
          ${state.kbs.map((kb) => `<option value="${kb.id}">${escapeHtml(kb.name)}</option>`).join('')}
        </select>
      </div>
      <div class="field">
        <label for="ds-desc">描述（可选）</label>
        <input id="ds-desc" type="text" placeholder="这批问题来自哪里" maxlength="500" />
      </div>`,
    footer: '<button class="btn" data-cancel>取消</button><button class="btn primary" data-save>创建</button>',
    onMount: (root, close) => {
      $('#ds-name', root).focus();
      $('[data-save]', root).addEventListener('click', async () => {
        const name = $('#ds-name', root).value.trim();
        if (!name) {
          toast('请输入评测集名称', 'warn');
          return;
        }
        try {
          const dataset = await api('/eval/datasets', {
            method: 'POST',
            json: {
              name,
              kb_id: $('#ds-kb', root).value,
              description: $('#ds-desc', root).value.trim(),
            },
          });
          close();
          state.datasets.unshift(dataset);
          state.datasetId = dataset.id;
          renderDatasetList();
          await renderDatasetView();
          toast('评测集已创建', 'ok');
        } catch (error) {
          toast(error.message, 'err');
        }
      });
      $('[data-cancel]', root).addEventListener('click', close);
    },
  });
}

async function openCaseModal() {
  const dataset = currentDataset();
  if (!dataset) return;

  let docs = [];
  try {
    docs = await api(`/knowledge-bases/${dataset.kb_id}/documents`);
  } catch { /* the checkbox list simply stays empty */ }
  const ready = docs.filter((doc) => doc.status === 'ready');

  openModal({
    title: '添加评测用例',
    body: `
      <div class="field">
        <label for="case-q">问题</label>
        <textarea id="case-q" placeholder="例如：E-201 设备的最大工作压力是多少？"></textarea>
      </div>
      <div class="field">
        <label for="case-kw">关键词（可选，用逗号分隔）</label>
        <input id="case-kw" type="text" placeholder="0.8MPa, 工作压力" />
      </div>
      <div class="field">
        <label for="case-chunks">参考分块 ID（可选，用逗号分隔）</label>
        <input id="case-chunks" type="text" placeholder="在文档「分块」预览里复制 ID，指标会更准" />
      </div>
      <div class="field">
        <label>参考文档（可选，可多选）</label>
        ${ready.length
          ? `<div class="check-list">${ready.map((doc) => `
              <label><input type="checkbox" value="${doc.id}" /><span>${escapeHtml(doc.name)}</span></label>`).join('')}
            </div>`
          : '<p class="faint" style="font-size:12.5px">该知识库还没有已就绪的文档</p>'}
      </div>
      <p class="faint" style="font-size:12px">关键词 / 参考分块 / 参考文档至少填写一项。分块级标注最准，文档级标注会把同文档的所有分块都算作相关。</p>`,
    footer: '<button class="btn" data-cancel>取消</button><button class="btn primary" data-save>保存</button>',
    onMount: (root, close) => {
      $('#case-q', root).focus();
      $('[data-cancel]', root).addEventListener('click', close);
      $('[data-save]', root).addEventListener('click', async () => {
        const question = $('#case-q', root).value.trim();
        const keywords = $('#case-kw', root).value
          .split(/[,，;；]/)
          .map((item) => item.trim())
          .filter(Boolean);
        const chunkIds = $('#case-chunks', root).value
          .split(/[,，;；\s]+/)
          .map((item) => item.trim())
          .filter(Boolean);
        const reference = $$('input[type="checkbox"]:checked', root).map((input) => input.value);

        if (!question) {
          toast('请输入问题', 'warn');
          return;
        }
        if (!keywords.length && !reference.length && !chunkIds.length) {
          toast('请至少填写关键词、参考分块或选择参考文档', 'warn');
          return;
        }

        try {
          const created = await api(`/eval/datasets/${dataset.id}/cases`, {
            method: 'POST',
            json: {
              question,
              keywords,
              reference_doc_ids: reference,
              reference_chunk_ids: chunkIds,
            },
          });
          close();
          state.cases.push(created);
          renderCaseTable();
          toggleRunButton();
          refreshDatasetMeta();
          toast('用例已添加', 'ok');
        } catch (error) {
          toast(error.message, 'err');
        }
      });
    },
  });
}

/* ---------------------------------------------------------------- routing */
function clearRunTimer() {
  if (state.runTimer) {
    clearTimeout(state.runTimer);
    state.runTimer = null;
  }
}

function route() {
  const name = (location.hash.replace(/^#\/?/, '').split('/')[0] || 'documents').toLowerCase();
  state.route = name === 'evaluate' ? 'evaluate' : 'documents';

  $$('.tabs a').forEach((link) => {
    link.classList.toggle('active', link.dataset.tab === state.route);
  });

  clearDocTimer();
  clearRunTimer();

  const render = state.route === 'documents' ? renderDocumentsPage : renderEvaluationPage;
  render().catch((error) => {
    $('#view').innerHTML = `<div class="card">${emptyState({
      icon: '!',
      title: '页面加载失败',
      text: error.message,
    })}</div>`;
  });
}

loadHealth();
window.addEventListener('hashchange', route);
route();
