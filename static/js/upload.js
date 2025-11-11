document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const resultBox = document.getElementById('resultBox');
    const predictionText = document.getElementById('predictionText');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const submitBtn = document.getElementById('submitBtn');
    const wasteImageInput = document.getElementById('wasteImage');

    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault(); // Stop the default form submission (page reload)

            // 1. Reset state
            resultBox.classList.add('hidden');
            predictionText.textContent = '';
            
            // Basic validation
            if (wasteImageInput.files.length === 0) {
                alert('Please select an image file to classify.');
                return;
            }

            // 2. Prepare data and UI for processing
            const formData = new FormData(form);
            submitBtn.disabled = true; // Disable button to prevent multiple submissions
            loadingIndicator.classList.remove('hidden'); // Show loading indicator

            // 3. Send data to Flask server
            fetch('/upload.html', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // 4. Handle response from Flask
                if (data.success) {
                    predictionText.textContent = `Classification: ${data.prediction}`;
                    resultBox.classList.remove('hidden');
                } else {
                    // Display server-side error (e.g., "Invalid file type")
                    predictionText.textContent = `Error: ${data.error || 'Classification failed.'}`;
                    resultBox.classList.remove('hidden');
                }
            })
            .catch(error => {
                // 5. Handle network or other errors
                console.error('Error during fetch operation:', error);
                predictionText.textContent = 'An unexpected error occurred. Check console for details.';
                resultBox.classList.remove('hidden');
            })
            .finally(() => {
                // 6. Restore UI state
                submitBtn.disabled = false;
                loadingIndicator.classList.add('hidden');
            });
        });
    }
});