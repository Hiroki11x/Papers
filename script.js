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

    issuesContainer.innerHTML = "<h1>Loading...</h1>"; // 表示をクリア

    if (hash.startsWith("#/issues/")) {
        const issueNumber = hash.split("/")[2];
        await renderDetailView(issueNumber);
    } else {
        const page = parseInt(params.get("page") || "1", 10);
        const selectedLabel = params.get("label") || null;
        await renderListView(page, selectedLabel);
    }
}

// --- API Fetching -------------------------------------
// Issue一覧を取得
async function fetchIssues(page, label) {
    let url = `${apiBaseUrl}/issues?page=${page}&per_page=${perPage}&state=open`;
    if (label) {
        url += `&labels=${encodeURIComponent(label)}`;
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`GitHub API error: ${response.status}`);
    const issues = await response.json();
    const linkHeader = response.headers.get("Link");
    return { issues, linkHeader };
}

// Issue詳細を取得
async function fetchIssue(issueNumber) {
    const response = await fetch(`${apiBaseUrl}/issues/${issueNumber}`);
    if (!response.ok) throw new Error(`GitHub API error: ${response.status}`);
    return await response.json();
}

// 全てのラベルを取得
async function fetchLabels() {
    const response = await fetch(`${apiBaseUrl}/labels`);
    if (!response.ok) return []; // ラベルがなくてもエラーにしない
    return await response.json();
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
            issuesContainer.innerHTML += "<p>No issues found for this filter.</p>";
        } else {
            const listElement = document.createElement("div");
            issues.forEach(issue => listElement.appendChild(createIssueCard(issue)));
            issuesContainer.appendChild(listElement);
        }

        // ページネーションをレンダリング
        renderPagination(page, linkHeader, selectedLabel);

    } catch (error) {
        issuesContainer.innerHTML = `<p>Error loading issues: ${error.message}. Please check if the repository is public and has issues.</p>`;
    }
}

// 詳細表示をレンダリング
async function renderDetailView(issueNumber) {
    try {
        const issue = await fetchIssue(issueNumber);
        issuesContainer.innerHTML = ""; // コンテナをクリア

        const backLink = document.createElement("a");
        backLink.href = "javascript:history.back()"; // 戻る機能
        backLink.classList.add("back-link");
        backLink.textContent = "← Back to list";

        const titleElement = document.createElement("h2");
        titleElement.textContent = issue.title;

        const metaElement = createMetaElement(issue);
        const bodyElement = createBodyElement(issue);

        issuesContainer.appendChild(backLink);
        issuesContainer.appendChild(titleElement);
        issuesContainer.appendChild(metaElement);
        issuesContainer.appendChild(bodyElement);

    } catch (error) {
        issuesContainer.innerHTML = `<p>Error loading issue: ${error.message}.</p>`;
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

    issueCard.appendChild(titleElement);
    issueCard.appendChild(metaElement);
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
            // Changed to add brackets and space
            tagElement.textContent = `[${label.name}]`;
            tagsElement.appendChild(tagElement);
        });
        metaElement.appendChild(tagsElement);
    }
    return metaElement;
}

// Issue本文の要素を作成
function createBodyElement(issue) {
    const bodyElement = document.createElement("div");
    bodyElement.classList.add("issue-body");
    // Note: For a real app, use a Markdown parser like 'marked' to safely render HTML
    bodyElement.innerHTML = `<pre>${issue.body || "No description provided."}</pre>`;
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
            link.href = `?${params.toString()}`;
        } else {
            link.classList.add("disabled");
        }
        if (page === currentPage) {
            link.classList.add("current");
        }
        return link;
    };

    paginationContainer.appendChild(createPageLink(links.prev, "« Previous"));

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

    paginationContainer.appendChild(createPageLink(links.next, "Next »"));
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