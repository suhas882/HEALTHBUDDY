#script.js
const sections = [
  { btn: "readMoreBtn1", content: "moreContent1" },
  { btn: "readMoreBtn2", content: "moreContent2" },
  { btn: "readMoreBtn3", content: "moreContent3" },
];

sections.forEach(({ btn, content }) => {
  const button = document.getElementById(btn);
  const moreContent = document.getElementById(content);

  button.addEventListener("click", () => {
    const isHidden = moreContent.style.display === "none";

    moreContent.style.display = isHidden ? "block" : "none";
    button.textContent = isHidden ? "Read Less" : "Read More";
  });
});


function goToSection4() {
  document.getElementById("section1").style.display = "none";
  document.getElementById("section2").style.display = "none";
  document.getElementById("section3").style.display = "none";
  document.getElementById("section4").style.display = "block";
}

function backToHome() {
  document.getElementById("section1").style.display = "block";
  document.getElementById("section2").style.display = "block";
  document.getElementById("section3").style.display = "block";
  document.getElementById("section4").style.display = "none";
}

window.onload = () => {
  document.getElementById("section1").style.display = "block";
  document.getElementById("section2").style.display = "block";
  document.getElementById("section3").style.display = "block";

  document.getElementById("send-btn").addEventListener("click", sendMessage);
  document.getElementById("user-input").addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
  });
};

function scrollToBottom() {
  const messagesDiv = document.getElementById("chat-messages");
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addMessage(text, sender) {
  const messagesDiv = document.getElementById("chat-messages");
  const messageElement = document.createElement("div");

  const formattedText = text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");

  messageElement.innerHTML = formattedText;
  messageElement.classList.add("message", `${sender}-message`);
  messagesDiv.appendChild(messageElement);

  scrollToBottom();
}

function sendMessage() {
  const inputField = document.getElementById("user-input");
  const message = inputField.value.trim();

  if (!message) return;

  addMessage(message, "user");
  inputField.value = "";

  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  })
    .then((response) => {
      if (!response.ok) throw new Error("Network Error: " + response.statusText);
      return response.json();
    })
    .then((data) => {
      addMessage(data.reply, "bot");
    })
    .catch((error) => {
      console.error("Error:", error);
      addMessage("Error connecting to the server. Please check backend.", "bot");
    });
}
