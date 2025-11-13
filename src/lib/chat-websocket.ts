export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: number;
}

export interface ChatWebSocketCallbacks {
  onProgress?: (chunk: string) => void;
  onEnd?: () => void;
  onError?: (error?: string) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export class ChatWebSocket {
  private ws: WebSocket | null = null;
  private callbacks: ChatWebSocketCallbacks = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectDelay = 1000;
  private isConnecting = false;
  private sessionId: string;

  constructor(private endpoint: string, sessionId?: string) {
    // Generate session ID if not provided
    this.sessionId = sessionId || this.generateSessionId();
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
  }

  getSessionId(): string {
    return this.sessionId;
  }

  connect(callbacks: ChatWebSocketCallbacks = {}): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
        resolve();
        return;
      }

      this.isConnecting = true;
      this.callbacks = callbacks;

      try {
        // Add session_id as query parameter
        const wsUrl = `${this.endpoint}?session_id=${this.sessionId}`;
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.callbacks.onConnect?.();
          resolve();
        };

        this.ws.onmessage = (event) => {
          const data = event.data;
          
          if (data === '<<<END>>>') {
            this.callbacks.onEnd?.();
          } else if (data === '<<<ERROR>>>') {
            this.callbacks.onError?.('Server error occurred');
          } else {
            this.callbacks.onProgress?.(data);
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected', event.code, event.reason);
          this.isConnecting = false;
          this.callbacks.onDisconnect?.();
          
          // Auto-reconnect on unexpected closure
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            setTimeout(() => {
              this.reconnectAttempts++;
              this.connect(this.callbacks);
            }, this.reconnectDelay * this.reconnectAttempts);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
          this.callbacks.onError?.('Connection error');
          reject(error);
        };

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  sendMessage(
    message: string, 
    history: ChatMessage[] = [],
    callbacks: Omit<ChatWebSocketCallbacks, 'onConnect' | 'onDisconnect'> = {}
  ): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      callbacks.onError?.('WebSocket not connected');
      return;
    }

    // Override callbacks for this specific message
    this.callbacks = { ...this.callbacks, ...callbacks };

    const payload = {
      message,
      history
    };

    try {
      this.ws.send(JSON.stringify(payload));
    } catch (error) {
      console.error('Failed to send message:', error);
      callbacks.onError?.('Failed to send message');
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  get connectionState(): number | null {
    return this.ws?.readyState ?? null;
  }
}