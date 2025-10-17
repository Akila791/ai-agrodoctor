// AI AgroDoctor - Connected to Python Backend
class AgroDoctor {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.form = document.getElementById('plantDiagnosisForm');
        this.resultDiv = document.getElementById('diagnosisResult');
        this.loadingDiv = document.getElementById('loadingSpinner');
        this.imageUpload = document.getElementById('imageUpload');
        this.imagePreview = document.getElementById('imagePreview');
        this.modelStatus = document.getElementById('modelStatus');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        
        this.initializeEventListeners();
        this.checkModelStatus();
    }

    initializeEventListeners() {
        this.form.addEventListener('submit', (e) => this.handleFormSubmit(e));
        this.imageUpload.addEventListener('change', (e) => this.handleImageUpload(e));
        
        const newDiagnosisBtn = document.getElementById('newDiagnosisBtn');
        if (newDiagnosisBtn) {
            newDiagnosisBtn.addEventListener('click', () => this.resetForm());
        }
    }

    async checkModelStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/health`);
            const data = await response.json();
            
            if (data.model_loaded) {
                this.statusIndicator.textContent = '✅';
                this.statusText.textContent = 'AI Model Ready - High Accuracy Predictions Available';
                this.statusIndicator.className = 'status-indicator status-ready';
            } else {
                this.statusIndicator.textContent = '⚠️';
                this.statusText.textContent = 'AI Model Not Available - Using Symptom Analysis Only';
                this.statusIndicator.className = 'status-indicator status-warning';
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            this.statusIndicator.textContent = '❌';
            this.statusText.textContent = 'Backend Connection Error';
            this.statusIndicator.className = 'status-indicator status-error';
        }
    }

    async handleFormSubmit(e) {
        e.preventDefault();
        
        const formData = new FormData(this.form);
        const plantType = formData.get('plantType');
        const symptoms = formData.get('symptoms');
        const environment = formData.get('environment');
        const imageFile = formData.get('imageUpload');
        
        this.showLoading();
        
        try {
            // Prepare request data
            const requestData = {
                plantType: plantType,
                symptoms: symptoms,
                environment: environment,
                imageData: null
            };
            
            // Convert image to base64 if provided
            if (imageFile && imageFile.size > 0) {
                requestData.imageData = await this.convertImageToBase64(imageFile);
            }
            
            // Send request to Python backend
            const response = await fetch(`${this.apiBaseUrl}/api/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.message);
            }
            
            // Display results
            this.displayDiagnosis(result);
            
        } catch (error) {
            console.error('Error during analysis:', error);
            this.showError('Analysis failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    convertImageToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    displayDiagnosis(result) {
        // Update plant name
        document.getElementById('plantNameText').textContent = result.plantName || 'Plant Species / ශාක විශේෂය';
        
        // Update disease text
        document.getElementById('diseaseText').textContent = result.disease || 'Unknown Condition';
        
        // Update confidence level with percentage
        const confidenceDiv = document.getElementById('confidenceLevel');
        const confidencePercent = Math.round(result.confidence || 0);
        confidenceDiv.textContent = `${confidencePercent}%`;
        
        // Set confidence class based on percentage
        if (confidencePercent >= 80) {
            confidenceDiv.className = 'confidence-badge confidence-high';
        } else if (confidencePercent >= 60) {
            confidenceDiv.className = 'confidence-badge confidence-medium';
        } else {
            confidenceDiv.className = 'confidence-badge confidence-low';
        }
        
        // Add uncertainty message for low confidence
        if (confidencePercent < 60) {
            const existingMsg = confidenceDiv.parentNode.querySelector('.uncertainty-message');
            if (!existingMsg) {
                const uncertaintyMsg = document.createElement('p');
                uncertaintyMsg.className = 'uncertainty-message';
                uncertaintyMsg.innerHTML = '<strong>⚠️ Need image or more symptom details for accurate diagnosis.</strong><br><span class="sinhala-text">නිරවද්‍ය විනිශ්චය සඳහා ඡායාරූපයක් හෝ වැඩි රෝග ලක්ෂණ විස්තර අවශ්‍යයි.</span>';
                confidenceDiv.parentNode.appendChild(uncertaintyMsg);
            }
        }
        
        // Update biological explanation
        let biologicalText = result.biologicalExplanation || 'No explanation available.';
        if (result.visualAnalysis) {
            biologicalText += `\n\nVisual Analysis: ${result.visualAnalysis}`;
        }
        document.getElementById('biologicalText').textContent = biologicalText;
        
        // Update treatments
        const treatmentsList = document.getElementById('treatmentsList');
        treatmentsList.innerHTML = '';
        if (result.treatments && result.treatments.length > 0) {
            result.treatments.forEach(treatment => {
                const li = document.createElement('li');
                li.textContent = treatment;
                treatmentsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No specific treatments available. Consult with agricultural experts.';
            treatmentsList.appendChild(li);
        }
        
        // Update prevention tips
        const preventionList = document.getElementById('preventionList');
        preventionList.innerHTML = '';
        if (result.prevention && result.prevention.length > 0) {
            result.prevention.forEach(tip => {
                const li = document.createElement('li');
                li.textContent = tip;
                preventionList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'Maintain general plant health practices.';
            preventionList.appendChild(li);
        }
        
        // Show model information
        const modelInfo = document.getElementById('modelInfo');
        const modelUsed = document.getElementById('modelUsed');
        if (result.modelUsed) {
            modelUsed.textContent = `Analysis Method: ${result.modelUsed}`;
            modelInfo.style.display = 'block';
        } else {
            modelInfo.style.display = 'none';
        }
        
        // Show result
        this.resultDiv.style.display = 'block';
        this.resultDiv.scrollIntoView({ behavior: 'smooth' });
    }

    showError(message) {
        // Create error display
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `
            <h3>❌ Error</h3>
            <p>${message}</p>
            <p>Please try again or contact support if the problem persists.</p>
        `;
        
        // Insert before result div
        this.resultDiv.parentNode.insertBefore(errorDiv, this.resultDiv);
        this.resultDiv.style.display = 'none';
        
        // Auto-remove error after 10 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 10000);
    }

    handleImageUpload(e) {
        const file = e.target.files[0];
        if (file) {
            // Validate file type
            if (!file.type.startsWith('image/')) {
                alert('Please select a valid image file.');
                return;
            }
            
            // Validate file size (max 10MB)
            if (file.size > 10 * 1024 * 1024) {
                alert('Image file is too large. Please select an image smaller than 10MB.');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = document.createElement('img');
                img.src = event.target.result;
                img.alt = 'Plant image';
                img.style.maxWidth = '300px';
                img.style.maxHeight = '300px';
                img.style.borderRadius = '10px';
                img.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.1)';
                
                this.imagePreview.innerHTML = '';
                this.imagePreview.appendChild(img);
            };
            reader.readAsDataURL(file);
        }
    }

    showLoading() {
        this.loadingDiv.style.display = 'block';
        this.resultDiv.style.display = 'none';
        
        // Remove any existing error messages
        const errorMessages = document.querySelectorAll('.error-message');
        errorMessages.forEach(msg => msg.remove());
    }

    hideLoading() {
        this.loadingDiv.style.display = 'none';
    }

    resetForm() {
        this.form.reset();
        this.resultDiv.style.display = 'none';
        this.imagePreview.innerHTML = '';
        
        // Remove any existing error messages
        const errorMessages = document.querySelectorAll('.error-message');
        errorMessages.forEach(msg => msg.remove());
        
        this.form.scrollIntoView({ behavior: 'smooth' });
    }
}

// Initialize the AI AgroDoctor when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new AgroDoctor();
});

// Add some interactive features
document.addEventListener('DOMContentLoaded', () => {
    // Add smooth scrolling for better UX
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add form validation feedback
    const form = document.getElementById('plantDiagnosisForm');
    if (form) {
        const inputs = form.querySelectorAll('input, textarea, select');
        
        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                if (input.checkValidity()) {
                    input.style.borderColor = '#4a7c59';
                } else {
                    input.style.borderColor = '#dc3545';
                }
            });
        });
    }
});
