// Main application JavaScript
class VibeVoiceApp {
    constructor() {
        this.voices = [];
        this.currentRequest = null;
        this.isGenerating = false;
        this.websocketClient = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.checkApiStatus();
        this.loadVoices();
    }
    
    initializeElements() {
        // Input elements
        this.scriptInput = document.getElementById('script-input');
        this.charCount = document.getElementById('char-count');
        this.voiceSelects = [
            document.getElementById('speaker-1'),
            document.getElementById('speaker-2'),
            document.getElementById('speaker-3'),
            document.getElementById('speaker-4')
        ];
        
        // Settings elements
        this.cfgScale = document.getElementById('cfg-scale');
        this.cfgValue = document.getElementById('cfg-value');
        this.inferenceSteps = document.getElementById('inference-steps');
        this.stepsValue = document.getElementById('steps-value');
        this.sampleRate = document.getElementById('sample-rate');
        this.seedInput = document.getElementById('seed');
        
        // Control elements
        this.modeRadios = document.querySelectorAll('input[name="mode"]');
        this.generateBtn = document.getElementById('generate-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.retryBtn = document.getElementById('retry-btn');
        
        // Status elements
        this.apiStatus = document.getElementById('api-status');
        this.wsStatus = document.getElementById('ws-status');
        this.generationStatus = document.getElementById('generation-status');
        
        // Output elements
        this.progressSection = document.getElementById('progress-section');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        this.audioSection = document.getElementById('audio-section');
        this.audioPlayer = document.getElementById('audio-player');
        this.audioDuration = document.getElementById('audio-duration');
        this.audioFormat = document.getElementById('audio-format');
        this.downloadBtn = document.getElementById('download-btn');
        this.playStreamingBtn = document.getElementById('play-streaming-btn');
        this.errorSection = document.getElementById('error-section');
        this.errorText = document.getElementById('error-text');
        
        // Other elements
        this.spinner = document.getElementById('spinner');
    }
    
    setupEventListeners() {
        // Text input
        this.scriptInput.addEventListener('input', () => this.updateCharCount());
        
        // Settings sliders
        this.cfgScale.addEventListener('input', (e) => {
            this.cfgValue.textContent = e.target.value;
        });
        
        this.inferenceSteps.addEventListener('input', (e) => {
            this.stepsValue.textContent = e.target.value;
        });
        
        // Generation mode change
        this.modeRadios.forEach(radio => {
            radio.addEventListener('change', () => this.handleModeChange());
        });
        
        // Buttons
        this.generateBtn.addEventListener('click', () => this.startGeneration());
        this.stopBtn.addEventListener('click', () => this.stopGeneration());
        this.retryBtn.addEventListener('click', () => this.retryGeneration());
        this.downloadBtn.addEventListener('click', () => this.downloadAudio());
        this.playStreamingBtn.addEventListener('click', () => this.playStreamingAudio());
        
        // Audio player events
        this.audioPlayer.addEventListener('loadedmetadata', () => this.updateAudioInfo());
        this.audioPlayer.addEventListener('error', (e) => this.handleAudioError(e));
    }
    
    updateCharCount() {
        const count = this.scriptInput.value.length;
        this.charCount.textContent = count;
        
        if (count > 4500) {
            this.charCount.style.color = '#e53e3e';
        } else if (count > 4000) {
            this.charCount.style.color = '#dd6b20';
        } else {
            this.charCount.style.color = '#718096';
        }
    }
    
    async checkApiStatus() {
        try {
            const response = await fetch('/readyz');
            if (response.ok) {
                const status = await response.json();
                this.updateApiStatus(status.ready ? 'ready' : 'not-ready', status);
            } else {
                this.updateApiStatus('error', { error: 'Failed to fetch status' });
            }
        } catch (error) {
            console.error('API status check failed:', error);
            this.updateApiStatus('error', { error: error.message });
        }
    }
    
    updateApiStatus(status, data) {
        this.apiStatus.className = 'status-value';
        
        switch (status) {
            case 'ready':
                this.apiStatus.classList.add('status-ready');
                this.apiStatus.textContent = `Ready (${data.device})`;
                break;
            case 'not-ready':
                this.apiStatus.classList.add('status-warning');
                this.apiStatus.textContent = 'Model Loading...';
                break;
            case 'error':
                this.apiStatus.classList.add('status-error');
                this.apiStatus.textContent = 'Error';
                break;
            default:
                this.apiStatus.classList.add('status-unknown');
                this.apiStatus.textContent = 'Unknown';
        }
    }
    
    async loadVoices() {
        try {
            const response = await fetch('/api/voices');
            if (response.ok) {
                const data = await response.json();
                this.voices = data.voices;
                this.populateVoiceSelects();
            } else {
                console.error('Failed to load voices:', response.statusText);
                this.showError('Failed to load available voices');
            }
        } catch (error) {
            console.error('Voice loading error:', error);
            this.showError('Failed to load available voices: ' + error.message);
        }
    }
    
    populateVoiceSelects() {
        this.voiceSelects.forEach(select => {
            // Clear existing options except the first
            while (select.children.length > 1) {
                select.removeChild(select.lastChild);
            }
            
            // Add voice options
            this.voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.id;
                const displayName = `${voice.name} (${voice.language || 'Unknown'}, ${voice.gender || 'N/A'})`;
                option.textContent = displayName;
                select.appendChild(option);
            });
        });
    }
    
    handleModeChange() {
        const selectedMode = document.querySelector('input[name="mode"]:checked').value;
        const isStreaming = selectedMode === 'streaming';
        
        if (isStreaming) {
            this.playStreamingBtn.style.display = 'inline-block';
            this.updateWebSocketStatus('disconnected');
        } else {
            this.playStreamingBtn.style.display = 'none';
            this.updateWebSocketStatus('n/a');
            if (this.websocketClient) {
                this.websocketClient.disconnect();
            }
        }
    }
    
    updateWebSocketStatus(status) {
        this.wsStatus.className = 'status-value';
        
        switch (status) {
            case 'connecting':
                this.wsStatus.classList.add('ws-connecting');
                this.wsStatus.textContent = 'Connecting...';
                break;
            case 'connected':
                this.wsStatus.classList.add('ws-connected');
                this.wsStatus.textContent = 'Connected';
                break;
            case 'disconnected':
                this.wsStatus.classList.add('status-error');
                this.wsStatus.textContent = 'Disconnected';
                break;
            case 'error':
                this.wsStatus.classList.add('ws-error');
                this.wsStatus.textContent = 'Error';
                break;
            case 'n/a':
                this.wsStatus.classList.add('status-unknown');
                this.wsStatus.textContent = 'N/A';
                break;
        }
    }
    
    validateInput() {
        const script = this.scriptInput.value.trim();
        if (!script) {
            this.showError('Please enter some text to generate speech.');
            return false;
        }
        
        if (script.length > 5000) {
            this.showError('Script is too long. Maximum 5000 characters allowed.');
            return false;
        }
        
        const selectedVoices = this.getSelectedVoices();
        if (selectedVoices.length === 0) {
            this.showError('Please select at least one voice.');
            return false;
        }
        
        return true;
    }
    
    getSelectedVoices() {
        return this.voiceSelects
            .map(select => select.value)
            .filter(value => value !== '');
    }
    
    buildGenerationRequest() {
        const selectedVoices = this.getSelectedVoices();
        
        return {
            script: this.scriptInput.value.trim(),
            speakers: selectedVoices,
            cfg_scale: parseFloat(this.cfgScale.value),
            inference_steps: parseInt(this.inferenceSteps.value),
            sample_rate: parseInt(this.sampleRate.value),
            format: 'wav',
            seed: this.seedInput.value ? parseInt(this.seedInput.value) : null
        };
    }
    
    async startGeneration() {
        if (!this.validateInput()) {
            return;
        }
        
        const selectedMode = document.querySelector('input[name="mode"]:checked').value;
        
        if (selectedMode === 'streaming') {
            this.startStreamingGeneration();
        } else {
            this.startBlockingGeneration();
        }
    }
    
    async startBlockingGeneration() {
        this.setGenerating(true);
        this.hideError();
        this.showProgress('Preparing generation...');
        
        try {
            const request = this.buildGenerationRequest();
            this.currentRequest = request;
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
            
            this.stopBtn.onclick = () => {
                controller.abort();
                this.stopGeneration();
            };
            
            this.progressText.textContent = 'Generating audio...';
            this.setProgress(10);
            
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'audio/wav'
                },
                body: JSON.stringify(request),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || `HTTP ${response.status}`);
            }
            
            this.setProgress(80);
            this.progressText.textContent = 'Processing audio...';
            
            const audioBlob = await response.blob();
            this.setProgress(100);
            
            const audioUrl = URL.createObjectURL(audioBlob);
            this.audioPlayer.src = audioUrl;
            
            this.showAudioSection();
            this.hideProgress();
            this.setGenerating(false);
            
            this.updateGenerationStatus('completed');
            
        } catch (error) {
            console.error('Generation failed:', error);
            this.handleGenerationError(error);
        }
    }
    
    startStreamingGeneration() {
        if (!this.websocketClient) {
            this.websocketClient = new VibeVoiceWebSocket();
            
            this.websocketClient.onStatusChange = (status) => {
                this.updateWebSocketStatus(status);
            };
            
            this.websocketClient.onProgress = (data) => {
                this.handleStreamingProgress(data);
            };
            
            this.websocketClient.onAudioChunk = (chunk) => {
                this.handleAudioChunk(chunk);
            };
            
            this.websocketClient.onComplete = (data) => {
                this.handleStreamingComplete(data);
            };
            
            this.websocketClient.onError = (error) => {
                this.handleGenerationError(error);
            };
        }
        
        this.setGenerating(true);
        this.hideError();
        this.showProgress('Connecting to streaming service...');
        
        const request = this.buildGenerationRequest();
        this.currentRequest = request;
        
        this.websocketClient.generate(request);
        
        this.stopBtn.onclick = () => {
            this.websocketClient.disconnect();
            this.stopGeneration();
        };
    }
    
    handleStreamingProgress(data) {
        if (data.type === 'init') {
            this.progressText.textContent = 'Streaming audio generation started...';
            this.setProgress(10);
        } else if (data.type === 'progress') {
            const progress = Math.min(90, 10 + (data.ms_emitted / 1000) * 2); // Rough progress estimation
            this.setProgress(progress);
            this.progressText.textContent = `Generated ${(data.ms_emitted / 1000).toFixed(1)}s of audio...`;
        }
    }
    
    handleAudioChunk(chunk) {
        // This would be handled by the WebSocket client for real-time playback
        // For now, we'll just show that streaming is working
        if (this.websocketClient && this.websocketClient.audioBuffer) {
            this.playStreamingBtn.style.display = 'inline-block';
        }
    }
    
    handleStreamingComplete(data) {
        this.setProgress(100);
        this.progressText.textContent = 'Streaming complete!';
        
        if (this.websocketClient && this.websocketClient.getAudioBlob) {
            const audioBlob = this.websocketClient.getAudioBlob();
            if (audioBlob) {
                const audioUrl = URL.createObjectURL(audioBlob);
                this.audioPlayer.src = audioUrl;
                this.showAudioSection();
            }
        }
        
        this.hideProgress();
        this.setGenerating(false);
        this.updateGenerationStatus('completed');
    }
    
    stopGeneration() {
        if (this.websocketClient) {
            this.websocketClient.disconnect();
        }
        
        this.setGenerating(false);
        this.hideProgress();
        this.updateGenerationStatus('cancelled');
    }
    
    retryGeneration() {
        this.hideError();
        this.startGeneration();
    }
    
    handleGenerationError(error) {
        console.error('Generation error:', error);
        this.setGenerating(false);
        this.hideProgress();
        
        let errorMessage = 'An unexpected error occurred.';
        
        if (error.name === 'AbortError') {
            errorMessage = 'Generation was cancelled.';
            this.updateGenerationStatus('cancelled');
        } else if (error.message) {
            errorMessage = error.message;
            this.updateGenerationStatus('error');
        } else {
            this.updateGenerationStatus('error');
        }
        
        this.showError(errorMessage);
    }
    
    setGenerating(generating) {
        this.isGenerating = generating;
        this.generateBtn.disabled = generating;
        this.stopBtn.disabled = !generating;
        
        if (generating) {
            this.generateBtn.classList.add('loading');
        } else {
            this.generateBtn.classList.remove('loading');
        }
    }
    
    updateGenerationStatus(status) {
        this.generationStatus.className = 'status-value';
        
        switch (status) {
            case 'ready':
                this.generationStatus.classList.add('status-ready');
                this.generationStatus.textContent = 'Ready';
                break;
            case 'generating':
                this.generationStatus.classList.add('status-warning');
                this.generationStatus.textContent = 'Generating...';
                break;
            case 'completed':
                this.generationStatus.classList.add('status-ready');
                this.generationStatus.textContent = 'Completed';
                break;
            case 'cancelled':
                this.generationStatus.classList.add('status-unknown');
                this.generationStatus.textContent = 'Cancelled';
                break;
            case 'error':
                this.generationStatus.classList.add('status-error');
                this.generationStatus.textContent = 'Error';
                break;
        }
    }
    
    showProgress(text) {
        this.progressText.textContent = text;
        this.progressSection.style.display = 'block';
        this.setProgress(0);
    }
    
    hideProgress() {
        this.progressSection.style.display = 'none';
    }
    
    setProgress(percent) {
        this.progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    }
    
    showAudioSection() {
        this.audioSection.style.display = 'block';
    }
    
    hideAudioSection() {
        this.audioSection.style.display = 'none';
    }
    
    showError(message) {
        this.errorText.textContent = message;
        this.errorSection.style.display = 'block';
    }
    
    hideError() {
        this.errorSection.style.display = 'none';
    }
    
    updateAudioInfo() {
        if (this.audioPlayer.duration) {
            const duration = this.audioPlayer.duration;
            const minutes = Math.floor(duration / 60);
            const seconds = Math.floor(duration % 60);
            this.audioDuration.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
        
        this.audioFormat.textContent = 'WAV';
    }
    
    handleAudioError(error) {
        console.error('Audio playback error:', error);
        this.showError('Failed to load generated audio');
    }
    
    downloadAudio() {
        if (this.audioPlayer.src) {
            const link = document.createElement('a');
            link.href = this.audioPlayer.src;
            link.download = `vibevoice_${Date.now()}.wav`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
    
    playStreamingAudio() {
        if (this.websocketClient && this.websocketClient.playStreaming) {
            this.websocketClient.playStreaming();
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.vibeVoiceApp = new VibeVoiceApp();
});