const owner = "hiroki11x";
const repo = "Papers";

const issuesContainer = document.getElementById("issues-container");

// ダミーのIssueデータ
const issues = [
  {
    title: "サンプルIssue 1",
    body: "これはサンプルIssue 1の本文です。",
  },
  {
    title: "サンプルIssue 2",
    body: "これはサンプルIssue 2の本文です。",
  },
];

function displayIssues() {
  for (const issue of issues) {
    const issueElement = document.createElement("div");
    issueElement.classList.add("issue");

    const titleElement = document.createElement("h2");
    titleElement.textContent = issue.title;

    const bodyElement = document.createElement("p");
    bodyElement.textContent = issue.body;

    issueElement.appendChild(titleElement);
    issueElement.appendChild(bodyElement);

    issuesContainer.appendChild(issueElement);
  }
}

displayIssues();