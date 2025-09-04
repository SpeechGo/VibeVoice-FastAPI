// WebSocket client for VibeVoice streaming
class VibeVoiceWebSocket {
    constructor() {
        this.ws = null;
        this.audioContext = null;
        this.audioBuffer = [];
        this.sampleRate = 24000;
        this.isConnected = false;
        this.requestId = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 3;
        
        // Callbacks
        this.onStatusChange = null;
        this.onProgress = null;
        this.onAudioChunk = null;
        this.onComplete = null;
        this.onError = null;
        
        // Audio playback
        this.audioSource = null;
        this.gainNode = null;
        this.isPlaying = false;
        
        this.initializeAudioContext();
    }
    
    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.gainNode = this.audioContext.createGain();
            this.gainNode.connect(this.audioContext.destination);
        } catch (error) {
            console.error('Failed to initialize audio context:', error);
        }
    }
    
    connect() {
        if (this.isConnected) {
            return Promise.resolve();
        }
        
        return new Promise((resolve, reject) => {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/generate`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.isConnected = true;
                    this.reconnectAttempts = 0;
                    this.notifyStatusChange('connected');
                    resolve();
                };
                
                this.ws.onmessage = (event) => {
                    this.handleMessage(event);
                };
                
                this.ws.onclose = (event) => {
                    console.log('WebSocket closed:', event.code, event.reason);
                    this.isConnected = false;
                    this.notifyStatusChange('disconnected');
                    
                    if (event.code === 1013) {
                        this.notifyError(new Error('Service is busy. Please try again later.'));
                    } else if (event.code === 1011) {
                        this.notifyError(new Error('Internal server error occurred.'));
                    } else if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                        this.attemptReconnect();
                    }
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.notifyStatusChange('error');
                    this.notifyError(new Error('WebSocket connection failed'));
                    reject(error);
                };
                
            } catch (error) {
                console.error('Failed to create WebSocket:', error);
                reject(error);
            }
        });
    }
    
    disconnect() {
        if (this.ws && this.isConnected) {
            this.ws.close(1000, 'Client disconnect');
        }
        this.cleanup();
    }
    
    cleanup() {
        this.isConnected = false;
        this.audioBuffer = [];
        this.requestId = null;
        
        if (this.audioSource) {
            try {
                this.audioSource.stop();
            } catch (e) {
                // Ignore if already stopped
            }
            this.audioSource = null;
        }
        
        this.isPlaying = false;
    }
    
    async attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.notifyError(new Error('Maximum reconnection attempts reached'));
            return;
        }
        
        this.reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
        
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms...`);
        
        setTimeout(() => {
            this.connect().catch(error => {
                console.error('Reconnection failed:', error);
            });
        }, delay);
    }
    
    async generate(request) {
        try {
            this.notifyStatusChange('connecting');
            await this.connect();
            
            // Add request ID if not present
            if (!request.request_id) {
                request.request_id = this.generateRequestId();
            }
            this.requestId = request.request_id;
            
            // Clear previous audio data
            this.audioBuffer = [];
            
            // Send generation request
            this.ws.send(JSON.stringify(request));
            
        } catch (error) {
            console.error('Generation request failed:', error);
            this.notifyError(error);
        }
    }
    
    handleMessage(event) {
        if (typeof event.data === 'string') {
            // Text message - JSON protocol
            try {
                const data = JSON.parse(event.data);
                this.handleJsonMessage(data);
            } catch (error) {
                console.error('Failed to parse JSON message:', error);
            }
        } else {
            // Binary message - PCM16 audio data
            this.handleBinaryMessage(event.data);
        }
    }
    
    handleJsonMessage(data) {
        switch (data.type) {
            case 'init':
                console.log('Streaming initialized:', data);
                this.sampleRate = data.sample_rate || 24000;
                this.notifyProgress(data);
                break;
                
            case 'progress':
                console.log('Streaming progress:', data);
                this.notifyProgress(data);
                break;
                
            case 'final':
                console.log('Streaming complete:', data);
                this.notifyComplete(data);
                break;
                
            case 'error':
                console.error('Streaming error:', data);
                this.notifyError(new Error(`${data.code}: ${data.message}`));
                break;
                
            default:
                console.warn('Unknown message type:', data.type);
        }
    }
    
    handleBinaryMessage(data) {
        // Convert ArrayBuffer to Int16Array (PCM16 little-endian)
        const pcm16Data = new Int16Array(data);
        
        // Store for later playback or download
        this.audioBuffer.push(pcm16Data);
        
        // Notify that we received an audio chunk
        this.notifyAudioChunk(pcm16Data);
        
        // If we want real-time playback, we could play chunks as they arrive
        if (this.isPlaying) {
            this.playAudioChunk(pcm16Data);
        }
    }
    
    async playAudioChunk(pcm16Data) {
        if (!this.audioContext || this.audioContext.state === 'suspended') {
            try {
                await this.audioContext.resume();
            } catch (error) {
                console.error('Failed to resume audio context:', error);
                return;
            }
        }
        
        try {
            // Convert PCM16 to float32 for Web Audio API
            const float32Data = new Float32Array(pcm16Data.length);
            for (let i = 0; i < pcm16Data.length; i++) {
                float32Data[i] = pcm16Data[i] / 32768.0; // Convert to [-1, 1] range
            }
            
            // Create audio buffer
            const audioBuffer = this.audioContext.createBuffer(1, float32Data.length, this.sampleRate);
            audioBuffer.getChannelData(0).set(float32Data);
            
            // Create and play audio source
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.gainNode);
            source.start();
            
        } catch (error) {
            console.error('Failed to play audio chunk:', error);
        }
    }
    
    playStreaming() {
        if (this.audioBuffer.length === 0) {
            console.warn('No audio buffer available for streaming playback');
            return;
        }
        
        this.isPlaying = true;
        
        // Play all buffered chunks
        this.audioBuffer.forEach((chunk, index) => {
            setTimeout(() => {
                if (this.isPlaying) {
                    this.playAudioChunk(chunk);
                }
            }, index * 100); // Slight delay between chunks
        });
    }
    
    stopStreaming() {
        this.isPlaying = false;
        
        if (this.audioSource) {
            try {
                this.audioSource.stop();
            } catch (e) {
                // Ignore if already stopped
            }
        }
    }
    
    getAudioBlob() {
        if (this.audioBuffer.length === 0) {
            return null;
        }
        
        try {
            // Calculate total length
            const totalLength = this.audioBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
            
            // Concatenate all chunks
            const concatenated = new Int16Array(totalLength);
            let offset = 0;
            
            for (const chunk of this.audioBuffer) {
                concatenated.set(chunk, offset);
                offset += chunk.length;
            }
            
            // Convert to WAV format
            const wavData = this.pcm16ToWav(concatenated, this.sampleRate);
            return new Blob([wavData], { type: 'audio/wav' });
            
        } catch (error) {
            console.error('Failed to create audio blob:', error);
            return null;
        }
    }
    
    pcm16ToWav(pcm16Data, sampleRate) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const byteRate = sampleRate * numChannels * bitsPerSample / 8;
        const blockAlign = numChannels * bitsPerSample / 8;
        const dataSize = pcm16Data.length * 2;
        const fileSize = 36 + dataSize;
        
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);
        
        // WAV header
        view.setUint32(0, 0x52494646, false); // 'RIFF'
        view.setUint32(4, fileSize, true);
        view.setUint32(8, 0x57415645, false); // 'WAVE'
        view.setUint32(12, 0x666d7420, false); // 'fmt '
        view.setUint32(16, 16, true); // PCM chunk size
        view.setUint16(20, 1, true); // PCM format
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        view.setUint32(36, 0x64617461, false); // 'data'
        view.setUint32(40, dataSize, true);
        
        // PCM data
        const pcmOffset = 44;
        for (let i = 0; i < pcm16Data.length; i++) {
            view.setInt16(pcmOffset + i * 2, pcm16Data[i], true);
        }
        
        return buffer;
    }
    
    generateRequestId() {
        return 'req_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    // Event notification methods
    notifyStatusChange(status) {
        if (this.onStatusChange) {
            this.onStatusChange(status);
        }
    }
    
    notifyProgress(data) {
        if (this.onProgress) {
            this.onProgress(data);
        }
    }
    
    notifyAudioChunk(chunk) {
        if (this.onAudioChunk) {
            this.onAudioChunk(chunk);
        }
    }
    
    notifyComplete(data) {
        if (this.onComplete) {
            this.onComplete(data);
        }
    }
    
    notifyError(error) {
        if (this.onError) {
            this.onError(error);
        }
    }
}

// Make it available globally
window.VibeVoiceWebSocket = VibeVoiceWebSocket;