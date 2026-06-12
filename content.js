// Content script to extract visible text from the webpage

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("SummarAIze content script received message:", message);

  if (message.action === "extractText") {
    try {
      // Extract all visible text from the webpage body
      const extractedText = document.body.innerText || "";
      
      // Send the extracted text back to the popup
      sendResponse({
        success: true,
        text: extractedText
      });
    } catch (error) {
      console.error("SummarAIze failed to extract text:", error);
      sendResponse({
        success: false,
        error: error.message
      });
    }
  }
  
  // Return true to keep the message channel open for sendResponse
  return true;
});
