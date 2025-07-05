const issuesContainer = document.getElementById("issues-container");

// ダミーのIssueデータ
const issues = [
  {
    title: "Attention Is All You Need",
    body: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    tags: ["NLP", "Transformer"],
    date: "2017-06-12"
  },
  {
    title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    body: "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
    tags: ["NLP", "BERT"],
    date: "2018-10-11"
  },
];

function displayIssues() {
  for (const issue of issues) {
    const issueCard = document.createElement("div");
    issueCard.classList.add("issue-card");

    const titleElement = document.createElement("h2");
    titleElement.textContent = issue.title;

    const metaElement = document.createElement("div");
    metaElement.classList.add("meta");
    metaElement.innerHTML = `
        <span>${issue.date}</span>
    `;

    const tagsElement = document.createElement("div");
    tagsElement.classList.add("tags");
    issue.tags.forEach(tag => {
        const tagElement = document.createElement("span");
        tagElement.textContent = tag;
        tagsElement.appendChild(tagElement);
    });
    metaElement.appendChild(tagsElement);


    const bodyElement = document.createElement("p");
    bodyElement.textContent = issue.body;

    issueCard.appendChild(titleElement);
    issueCard.appendChild(metaElement);
    issueCard.appendChild(bodyElement);

    issuesContainer.appendChild(issueCard);
  }
}

displayIssues();