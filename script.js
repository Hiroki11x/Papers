const owner = "Hiroki11x";
const repo = "Papers";
const apiBaseUrl = `https://api.github.com/repos/${owner}/${repo}`;
const issuesContainer = document.getElementById("issues-container");

// メインのルーター機能
async function router() {
    const hash = window.location.hash;
    issuesContainer.innerHTML = ""; // コンテナをクリア

    if (hash.startsWith("#/issues/")) {
        const issueNumber = hash.split("/")[2];
        await renderDetailView(issueNumber);
    } else {
        await renderListView();
    }
}

// 一覧表示をレンダリング
async function renderListView() {
    try {
        const response = await fetch(`${apiBaseUrl}/issues`);
        if (!response.ok) {
            throw new Error(`GitHub API error: ${response.status}`);
        }
        const issues = await response.json();

        if (issues.length === 0) {
            issuesContainer.innerHTML = "<p>No issues found.</p>";
            return;
        }

        for (const issue of issues) {
            const issueCard = document.createElement("a"); // aタグに変更
            issueCard.classList.add("issue-card");
            issueCard.href = `#/issues/${issue.number}`; // ハッシュリンクを設定

            const titleElement = document.createElement("h2");
            titleElement.textContent = issue.title;

            const metaElement = document.createElement("div");
            metaElement.classList.add("meta");
            metaElement.textContent = `Issue #${issue.number} opened on ${new Date(issue.created_at).toLocaleDateString()}`;

            issueCard.appendChild(titleElement);
            issueCard.appendChild(metaElement);
            issuesContainer.appendChild(issueCard);
        }
    } catch (error) {
        issuesContainer.innerHTML = `<p>Error loading issues: ${error.message}. Please check if the repository is public and has issues.</p>`;
    }
}

// 詳細表示をレンダリング
async function renderDetailView(issueNumber) {
    try {
        const response = await fetch(`${apiBaseUrl}/issues/${issueNumber}`);
        if (!response.ok) {
            throw new Error(`GitHub API error: ${response.status}`);
        }
        const issue = await response.json();

        const backLink = document.createElement("a");
        backLink.href = "#";
        backLink.classList.add("back-link");
        backLink.textContent = "← Back to list";

        const titleElement = document.createElement("h2");
        titleElement.textContent = issue.title;

        const metaElement = document.createElement("div");
        metaElement.classList.add("meta");
        metaElement.textContent = `Issue #${issue.number} opened on ${new Date(issue.created_at).toLocaleDateString()}`;

        const bodyElement = document.createElement("div");
        bodyElement.classList.add("issue-body");
        // GitHubのIssue BodyはMarkdown形式なので、preタグで整形して表示
        bodyElement.innerHTML = `<pre>${issue.body}</pre>`;

        issuesContainer.appendChild(backLink);
        issuesContainer.appendChild(titleElement);
        issuesContainer.appendChild(metaElement);
        issuesContainer.appendChild(bodyElement);

    } catch (error) {
        issuesContainer.innerHTML = `<p>Error loading issue: ${error.message}.</p>`;
    }
}

// イベントリスナーを設定
window.addEventListener("hashchange", router);
// 初期読み込み
router();