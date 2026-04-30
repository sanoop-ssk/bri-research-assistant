const puppeteer = require('puppeteer');

(async () => {
  console.log("Opening browser...");
  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();
  
  try {
    console.log("Visiting BRI DataLab...");
    // Go to the app and wait until the network is quiet (app is loaded)
    await page.goto('https://bri-datalab.streamlit.app', { waitUntil: 'networkidle2', timeout: 60000 });
    console.log("Successfully loaded the app to keep it awake!");
  } catch (error) {
    console.error("Failed to load:", error);
  } finally {
    await browser.close();
  }
})();
