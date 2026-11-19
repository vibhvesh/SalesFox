# SalesFox

A simple, voice-first retail product catalogue and billing prototype.

## Run locally
Run `python -m http.server 8000` in this folder and open http://localhost:8000. Camera and microphone access require HTTPS or localhost.

## Try it
1. Add products one at a time with a photo, name, unit and selling price.
2. Continue to billing and select a simulated shopkeeper.
3. Speak a command or type one, for example “Add two Tata Salt”.
4. Review the shared draft basket.
5. Press **Confirm & generate bill** to create the invoice. Nothing is invoiced automatically.
6. Print or save the invoice as PDF.

## GitHub Pages
In repository Settings → Pages, select **Deploy from a branch**, choose `main` and `/(root)`, then save.

## Prototype limitations
- Product photos, catalogue and draft basket are stored in this browser's localStorage.
- Five shopkeepers are simulated on one device; live multi-device Sanket PTT integration is not included yet.
- Voice uses the browser SpeechRecognition/Web Speech API where supported. Depending on the browser, recognition may use an online service; this is not local Whisper.
- Product matching is simple catalogue-name matching, not visual recognition.
- There is no shared server database, authentication or cross-device synchronization in this prototype.
