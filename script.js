console.log("Running script.js version 7");

const owner = "Hiroki11x";
const repo = "Papers";
const apiBaseUrl = `https://api.github.com/repos/${owner}/${repo}`;
const issuesContainer = document.getElementById("issues-container");
const perPage = 10; // 1ページあたりの表示件数

// --- Router -------------------------------------------
// URLを解析して適切なビューを表示するメイン機能
async function router() {
    const params = new URLSearchParams(window.location.search);
    const hash = window.location.hash;

    // ローディング表示
    issuesContainer.innerHTML = `
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <p class="loading-text">Loading...</p>
        </div>`;

    if (hash.startsWith("#/issues/")) {
        const issueNumber = hash.split("/")[2];
        await renderDetailView(issueNumber);
    } else {
        const page = parseInt(params.get("page") || "1", 10);
        const selectedLabel = params.get("label") || null;
        await renderListView(page, selectedLabel);
    }
}

// --- SPA Navigation -----------------------------------
// ページ遷移をSPAスタイルで処理
function navigateTo(url) {
    history.pushState(null, "", url);
    router();
}

// --- API Fetching -------------------------------------
// Issue一覧を取得
async function fetchIssues(page, label) {
    let url = `${apiBaseUrl}/issues?page=${page}&per_page=${perPage}&state=open`;
    if (label) {
        url += `&labels=${encodeURIComponent(label)}`;
    }
    const response = await fetch(url);
    if (response.status === 403 || response.status === 429) {
        throw new Error("GitHub API rate limit exceeded. Please wait a few minutes and try again.");
    }
    if (!response.ok) throw new Error(`GitHub API error: ${response.status}`);
    const issues = await response.json();
    const linkHeader = response.headers.get("Link");
    return { issues, linkHeader };
}

// Issue詳細を取得
async function fetchIssue(issueNumber) {
    const response = await fetch(`${apiBaseUrl}/issues/${issueNumber}`);
    if (response.status === 403 || response.status === 429) {
        throw new Error("GitHub API rate limit exceeded. Please wait a few minutes and try again.");
    }
    if (!response.ok) throw new Error(`GitHub API error: ${response.status}`);
    return await response.json();
}

// 全てのラベルを取得
async function fetchLabels() {
    const response = await fetch(`${apiBaseUrl}/labels`);
    if (!response.ok) return []; // ラベルがなくてもエラーにしない
    return await response.json();
}

// --- Markdown Parsing ---------------------------------
// 基本的なMarkdownをHTMLに変換
function parseMarkdown(markdown) {
    if (!markdown) return "<p>No description provided.</p>";

    // ブロック単位で処理するためにコードブロックを先に抽出
    const codeBlocks = [];
    let text = markdown.replace(/```(\w+)?\n([\s\S]*?)```/g, (_match, lang, code) => {
        const placeholder = `%%CODEBLOCK_${codeBlocks.length}%%`;
        codeBlocks.push(`<pre class="code-block ${lang || ''}"><code>${escapeHtml(code.trim())}</code></pre>`);
        return placeholder;
    });

    // 行ごとに処理
    const lines = text.split('\n');
    const outputBlocks = [];
    let currentList = null; // { type: 'ul' | 'ol', items: [] }
    let currentBlockquote = [];

    function flushList() {
        if (currentList) {
            const tag = currentList.type;
            outputBlocks.push(`<${tag}>${currentList.items.map(item => `<li>${item}</li>`).join('')}</${tag}>`);
            currentList = null;
        }
    }

    function flushBlockquote() {
        if (currentBlockquote.length > 0) {
            outputBlocks.push(`<blockquote>${currentBlockquote.join('<br>')}</blockquote>`);
            currentBlockquote = [];
        }
    }

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // コードブロックプレースホルダー
        if (line.trim().startsWith('%%CODEBLOCK_')) {
            flushList();
            flushBlockquote();
            outputBlocks.push(line.trim());
            continue;
        }

        // ブロック引用
        const blockquoteMatch = line.match(/^>\s?(.*)/);
        if (blockquoteMatch) {
            flushList();
            currentBlockquote.push(processInline(blockquoteMatch[1]));
            continue;
        } else {
            flushBlockquote();
        }

        // 見出し
        const h3Match = line.match(/^### (.*)/);
        if (h3Match) { flushList(); outputBlocks.push(`<h3>${processInline(h3Match[1])}</h3>`); continue; }
        const h2Match = line.match(/^## (.*)/);
        if (h2Match) { flushList(); outputBlocks.push(`<h2>${processInline(h2Match[1])}</h2>`); continue; }
        const h1Match = line.match(/^# (.*)/);
        if (h1Match) { flushList(); outputBlocks.push(`<h1>${processInline(h1Match[1])}</h1>`); continue; }

        // 水平線
        if (/^(---|\*\*\*)$/.test(line.trim())) {
            flushList();
            outputBlocks.push('<hr>');
            continue;
        }

        // 順序なしリスト
        const ulMatch = line.match(/^[\*\-\+]\s+(.*)/);
        if (ulMatch) {
            flushBlockquote();
            if (currentList && currentList.type === 'ul') {
                currentList.items.push(processInline(ulMatch[1]));
            } else {
                flushList();
                currentList = { type: 'ul', items: [processInline(ulMatch[1])] };
            }
            continue;
        }

        // 順序付きリスト
        const olMatch = line.match(/^\d+\.\s+(.*)/);
        if (olMatch) {
            flushBlockquote();
            if (currentList && currentList.type === 'ol') {
                currentList.items.push(processInline(olMatch[1]));
            } else {
                flushList();
                currentList = { type: 'ol', items: [processInline(olMatch[1])] };
            }
            continue;
        }

        // 空行
        if (line.trim() === '') {
            flushList();
            continue;
        }

        // 通常のテキスト行
        flushList();
        outputBlocks.push(`<p>${processInline(line)}</p>`);
    }

    flushList();
    flushBlockquote();

    let html = outputBlocks.join('\n');

    // コードブロックを復元
    codeBlocks.forEach((block, i) => {
        html = html.replace(`%%CODEBLOCK_${i}%%`, block);
        // <p>で囲まれてしまった場合の修正
        html = html.replace(`<p>${block}</p>`, block);
    });

    return html;
}

// インライン要素を処理
function processInline(text) {
    return text
        // インラインコード
        .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
        // 画像
        .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" class="markdown-image">')
        // リンク
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
        // 太字（**が先、*は後で処理）
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/__(.*?)__/g, '<strong>$1</strong>')
        // 斜体
        .replace(/\*([^\*]+)\*/g, '<em>$1</em>')
        .replace(/_([^_]+)_/g, '<em>$1</em>');
}

// HTMLエスケープ関数
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// --- Rendering ----------------------------------------
// 一覧表示をレンダリング
async function renderListView(page, selectedLabel) {
    try {
        const [{ issues, linkHeader }, allLabels] = await Promise.all([
            fetchIssues(page, selectedLabel),
            fetchLabels()
        ]);

        issuesContainer.innerHTML = ""; // コンテナをクリア

        // ラベルフィルターをレンダリング
        renderLabelFilter(allLabels, selectedLabel);

        // Issueカードをレンダリング
        if (issues.length === 0) {
            const emptyMsg = document.createElement("p");
            emptyMsg.classList.add("empty-message");
            emptyMsg.textContent = "No issues found for this filter.";
            issuesContainer.appendChild(emptyMsg);
        } else {
            const listElement = document.createElement("div");
            issues.forEach(issue => listElement.appendChild(createIssueCard(issue)));
            issuesContainer.appendChild(listElement);
        }

        // ページネーションをレンダリング
        renderPagination(page, linkHeader, selectedLabel);

    } catch (error) {
        issuesContainer.innerHTML = `<div class="error-message"><p>${escapeHtml(error.message)}</p></div>`;
    }
}

// 詳細表示をレンダリング
async function renderDetailView(issueNumber) {
    try {
        const issue = await fetchIssue(issueNumber);
        issuesContainer.innerHTML = ""; // コンテナをクリア

        const backLink = document.createElement("a");
        backLink.href = "?";
        backLink.classList.add("back-link");
        backLink.textContent = "\u2190 Back to list";
        backLink.addEventListener("click", (e) => {
            e.preventDefault();
            navigateTo("?");
        });

        // 詳細コンテンツをカードで包む
        const detailCard = document.createElement("div");
        detailCard.classList.add("detail-card");

        const titleElement = document.createElement("h2");
        titleElement.textContent = issue.title;
        titleElement.classList.add("issue-title");

        const metaElement = createMetaElement(issue);
        const bodyElement = createBodyElement(issue);

        detailCard.appendChild(titleElement);
        detailCard.appendChild(metaElement);
        detailCard.appendChild(bodyElement);

        issuesContainer.appendChild(backLink);
        issuesContainer.appendChild(detailCard);

    } catch (error) {
        issuesContainer.innerHTML = `<div class="error-message"><p>${escapeHtml(error.message)}</p></div>`;
    }
}

// --- UI Components ------------------------------------
// Issueカードを作成
function createIssueCard(issue) {
    const issueCard = document.createElement("a");
    issueCard.classList.add("issue-card");
    issueCard.href = `#/issues/${issue.number}`;

    const titleElement = document.createElement("h2");
    titleElement.textContent = issue.title;

    const metaElement = createMetaElement(issue);

    // プレビューテキストを追加
    const previewElement = createPreviewElement(issue);

    issueCard.appendChild(titleElement);
    issueCard.appendChild(metaElement);
    issueCard.appendChild(previewElement);
    return issueCard;
}

// Meta情報（日付、タグ）の要素を作成
function createMetaElement(issue) {
    const metaElement = document.createElement("div");
    metaElement.classList.add("meta");

    const dateElement = document.createElement("span");
    dateElement.classList.add("date");
    dateElement.textContent = `Opened on ${new Date(issue.created_at).toLocaleDateString()}`;
    metaElement.appendChild(dateElement);

    if (issue.labels && issue.labels.length > 0) {
        const tagsElement = document.createElement("div");
        tagsElement.classList.add("tags");
        issue.labels.forEach(label => {
            const tagElement = document.createElement("span");
            tagElement.classList.add("tag");
            tagElement.textContent = label.name;
            tagsElement.appendChild(tagElement);
        });
        metaElement.appendChild(tagsElement);
    }
    return metaElement;
}

// プレビューテキストの要素を作成
function createPreviewElement(issue) {
    const previewElement = document.createElement("div");
    previewElement.classList.add("issue-preview");

    if (issue.body) {
        // Markdownから画像とコードブロックを除去してプレビュー用のテキストを作成
        let previewText = issue.body
            .replace(/```[\s\S]*?```/g, '') // コードブロックを削除
            .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '') // 画像を削除
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1') // リンクはテキストのみ残す
            .replace(/[#*_`]/g, '') // マークダウン記号を削除
            .replace(/\n+/g, ' ') // 改行をスペースに
            .trim();

        // 150文字でカット
        if (previewText.length > 150) {
            previewText = previewText.substring(0, 150) + '...';
        }

        previewElement.textContent = previewText || "No description provided.";
    } else {
        previewElement.textContent = "No description provided.";
    }

    return previewElement;
}

// Issue本文の要素を作成
function createBodyElement(issue) {
    const bodyElement = document.createElement("div");
    bodyElement.classList.add("issue-body");

    // Markdownをパースして表示
    const parsedHTML = parseMarkdown(issue.body);
    bodyElement.innerHTML = parsedHTML;

    return bodyElement;
}

// ラベルフィルターのUIを作成
function renderLabelFilter(allLabels, selectedLabel) {
    const filterContainer = document.createElement("div");
    filterContainer.classList.add("label-filter");

    const createLabelLink = (name, text) => {
        const link = document.createElement("a");
        link.textContent = text;
        link.href = name ? `?label=${encodeURIComponent(name)}` : `?`;
        if (name === selectedLabel || (!name && !selectedLabel)) {
            link.classList.add("active");
        }
        // SPA ナビゲーション
        link.addEventListener("click", (e) => {
            e.preventDefault();
            navigateTo(link.href);
        });
        return link;
    };

    filterContainer.appendChild(createLabelLink(null, "All Issues"));
    allLabels.forEach(label => {
        filterContainer.appendChild(createLabelLink(label.name, label.name));
    });

    issuesContainer.appendChild(filterContainer);
}

// ページネーションのUIを作成
function renderPagination(currentPage, linkHeader, label) {
    const paginationContainer = document.createElement("div");
    paginationContainer.classList.add("pagination");

    const links = parseLinkHeader(linkHeader);
    const lastPage = links.last ? parseInt(new URL(links.last).searchParams.get("page"), 10) : currentPage;

    const createPageLink = (page, text = page) => {
        const link = document.createElement("a");
        link.textContent = text;
        if (page) {
            const params = new URLSearchParams();
            params.set("page", page);
            if (label) params.set("label", label);
            const url = `?${params.toString()}`;
            link.href = url;
            // SPA ナビゲーション
            link.addEventListener("click", (e) => {
                e.preventDefault();
                navigateTo(url);
            });
        } else {
            link.classList.add("disabled");
        }
        if (page === currentPage) {
            link.classList.add("current");
        }
        return link;
    };

    paginationContainer.appendChild(createPageLink(links.prev, "\u00ab Previous"));

    // Simplified pagination links
    for (let i = 1; i <= lastPage; i++) {
        if (i === 1 || i === lastPage || (i >= currentPage - 2 && i <= currentPage + 2)) {
            paginationContainer.appendChild(createPageLink(i));
        } else if (i === currentPage - 3 || i === currentPage + 3) {
            const span = document.createElement("span");
            span.textContent = "...";
            paginationContainer.appendChild(span);
        }
    }

    paginationContainer.appendChild(createPageLink(links.next, "Next \u00bb"));
    issuesContainer.appendChild(paginationContainer);
}

// --- Helpers ------------------------------------------
// GitHub APIのLinkヘッダーをパースする
function parseLinkHeader(header) {
    if (!header || header.length === 0) return {};
    const links = {};
    header.split(",").forEach(part => {
        const section = part.split(";");
        const url = section[0].replace(/<(.*)>/, "$1").trim();
        const name = section[1].replace(/rel="(.*)"/, "$1").trim();
        links[name] = url;
    });
    return links;
}

// --- Event Listeners ----------------------------------
window.addEventListener("popstate", router); // ブラウザの戻る/進むボタンに対応
window.addEventListener("hashchange", router);
document.addEventListener("DOMContentLoaded", router); // 初期読み込み
