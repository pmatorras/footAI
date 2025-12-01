
document.addEventListener("DOMContentLoaded", function() {
    // Wait a brief moment to ensure Dash layout is painted
    setTimeout(function() {
        console.log("Dash loaded, signaling parent...");
        // Send the secret handshake to the parent window
        window.parent.postMessage("DASH_APP_READY", "*");
    }, 1000);
});
