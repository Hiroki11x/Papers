const owner = "hiroki11x";
const repo = "Papers";

const issuesContainer = document.getElementById("issues-container");

async function getIssues() {
    const response = await fetch(`https://api.github.com/repos/${owner}/${repo}/issues`);
    const issues = await response.json();

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

getIssues();
