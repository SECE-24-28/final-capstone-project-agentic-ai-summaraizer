let mediaRecorder = null;
let recordedChunks = [];
let audioStream = null;
let audioCtx = null;

/**
 * Captures the audio stream of the active tab and starts recording.
 * Uses chrome.tabCapture.capture() to obtain the stream.
 * Connects the stream to the destination (speakers) so the user can still hear the tab's audio.
 * @returns {Promise<MediaStream>} Resolves with the captured MediaStream when recording starts.
 */
export function startRecording() {
  return new Promise((resolve, reject) => {
    recordedChunks = [];
    
    // Check if chrome.tabCapture is available (foreground context like popup/options)
    if (!chrome || !chrome.tabCapture || typeof chrome.tabCapture.capture !== 'function') {
      reject(new Error('chrome.tabCapture.capture is not available in this context.'));
      return;
    }

    chrome.tabCapture.capture({ audio: true, video: false }, (stream) => {
      if (!stream) {
        const err = chrome.runtime.lastError ? chrome.runtime.lastError.message : 'Unknown error';
        reject(new Error(`Failed to capture tab audio: ${err}`));
        return;
      }

      audioStream = stream;

      // Route audio back to the user's speakers so it doesn't mute the tab
      try {
        audioCtx = new AudioContext();
        const source = audioCtx.createMediaStreamSource(stream);
        source.connect(audioCtx.destination);
      } catch (err) {
        console.warn('Failed to redirect audio to destination speakers:', err);
      }

      // Initialize MediaRecorder to capture webm audio
      try {
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      } catch (err) {
        console.warn('MediaRecorder with audio/webm failed, falling back to default mimeType.', err);
        mediaRecorder = new MediaRecorder(stream);
      }

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunks.push(event.data);
        }
      };

      mediaRecorder.start();
      resolve(stream);
    });
  });
}

/**
 * Stops the active tab recording session and returns the recorded audio.
 * Releases all media tracks and closes the AudioContext.
 * @returns {Promise<Blob>} Resolves with the recorded audio Blob in audio/webm format.
 */
export function stopRecording() {
  return new Promise((resolve, reject) => {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      reject(new Error('No active recording session found.'));
      return;
    }

    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: 'audio/webm' });
      
      // Stop all tracks to release the stream capture lock
      if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
      }

      // Close AudioContext
      if (audioCtx && audioCtx.state !== 'closed') {
        audioCtx.close().catch(err => console.warn('Error closing AudioContext:', err));
        audioCtx = null;
      }

      resolve(blob);
    };

    mediaRecorder.stop();
  });
}
