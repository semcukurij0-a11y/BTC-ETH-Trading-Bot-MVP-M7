interface AlertConfig {
  telegram: {
    enabled: boolean;
    botToken: string;
    chatId: string;
  };
  email: {
    enabled: boolean;
    recipient: string;
  };
}

interface AlertEvent {
  type: 'order_fill' | 'error_spike' | 'daily_limit' | 'drawdown_breach' | 'heartbeat';
  severity: 'info' | 'warning' | 'critical';
  title: string;
  message: string;
  timestamp: string;
  data?: any;
}

export class AlertsService {
  private config: AlertConfig = {
    telegram: {
      enabled: true,
      botToken: '7552181112:AAHMSNJfbF2xfZ9tWPis4NddJGCfBPnFjvY',
      chatId: '8490393553'
    },
    email: {
      enabled: true,
      recipient: 'zombiewins23@gmail.com'
    }
  };

  private baseUrl = this.getApiBaseUrl();

  private getApiBaseUrl(): string {
    const envUrl = import.meta.env.VITE_API_BASE_URL;
    if (envUrl) {
      return envUrl;
    }
    return 'http://localhost:8000';
  }

  async sendAlert(event: AlertEvent): Promise<{ success: boolean; error?: string }> {
    try {
      const results = await Promise.allSettled([
        this.sendTelegramAlert(event),
        this.sendEmailAlert(event)
      ]);

      const telegramResult = results[0];
      const emailResult = results[1];

      const errors = [];
      if (telegramResult.status === 'rejected') {
        errors.push(`Telegram: ${telegramResult.reason}`);
      }
      if (emailResult.status === 'rejected') {
        errors.push(`Email: ${emailResult.reason}`);
      }

      if (errors.length > 0) {
        console.warn('Some alerts failed:', errors);
        return { success: false, error: errors.join('; ') };
      }

      return { success: true };
    } catch (error) {
      console.error('Error sending alerts:', error);
      return { success: false, error: `Failed to send alerts: ${error}` };
    }
  }

  private async sendTelegramAlert(event: AlertEvent): Promise<void> {
    if (!this.config.telegram.enabled) {
      console.log('Telegram alerts disabled');
      return;
    }

    const message = this.formatTelegramMessage(event);
    
    try {
      const response = await fetch(`https://api.telegram.org/bot${this.config.telegram.botToken}/sendMessage`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chat_id: this.config.telegram.chatId,
          text: message,
          parse_mode: 'HTML'
        })
      });

      if (!response.ok) {
        throw new Error(`Telegram API error: ${response.status}`);
      }

      console.log('✅ Telegram alert sent successfully');
    } catch (error) {
      console.error('❌ Failed to send Telegram alert:', error);
      throw error;
    }
  }

  private async sendEmailAlert(event: AlertEvent): Promise<void> {
    if (!this.config.email.enabled) {
      console.log('Email alerts disabled');
      return;
    }

    try {
      const response = await fetch(`${this.baseUrl}/alerts/email`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          recipient: this.config.email.recipient,
          subject: `Trading Bot Alert: ${event.title}`,
          message: this.formatEmailMessage(event),
          event_type: event.type,
          severity: event.severity,
          timestamp: event.timestamp
        })
      });

      if (!response.ok) {
        throw new Error(`Email API error: ${response.status}`);
      }

      console.log('✅ Email alert sent successfully');
    } catch (error) {
      console.error('❌ Failed to send email alert:', error);
      throw error;
    }
  }

  private formatTelegramMessage(event: AlertEvent): string {
    const emoji = this.getSeverityEmoji(event.severity);
    const time = new Date(event.timestamp).toLocaleString();
    
    return `${emoji} <b>${event.title}</b>\n\n` +
           `${event.message}\n\n` +
           `📅 Time: ${time}\n` +
           `🔔 Type: ${event.type.replace('_', ' ').toUpperCase()}\n` +
           `⚠️ Severity: ${event.severity.toUpperCase()}`;
  }

  private formatEmailMessage(event: AlertEvent): string {
    const time = new Date(event.timestamp).toLocaleString();
    
    return `
Trading Bot Alert

Title: ${event.title}
Message: ${event.message}

Event Details:
- Type: ${event.type.replace('_', ' ').toUpperCase()}
- Severity: ${event.severity.toUpperCase()}
- Time: ${time}

${event.data ? `Additional Data:\n${JSON.stringify(event.data, null, 2)}` : ''}

---
Trading Bot System
Generated at: ${new Date().toISOString()}
    `.trim();
  }

  private getSeverityEmoji(severity: string): string {
    switch (severity) {
      case 'critical': return '🚨';
      case 'warning': return '⚠️';
      case 'info': return 'ℹ️';
      default: return '📢';
    }
  }

  // Specific alert methods for different events
  async sendOrderFillAlert(orderData: any): Promise<void> {
    const event: AlertEvent = {
      type: 'order_fill',
      severity: 'info',
      title: 'Order Filled',
      message: `Order ${orderData.side} ${orderData.quantity} ${orderData.symbol} at ${orderData.price}`,
      timestamp: new Date().toISOString(),
      data: orderData
    };

    await this.sendAlert(event);
  }

  async sendErrorSpikeAlert(errorCount: number, timeWindow: string): Promise<void> {
    const event: AlertEvent = {
      type: 'error_spike',
      severity: 'critical',
      title: 'Error Spike Detected',
      message: `Circuit breaker triggered: ${errorCount} errors in ${timeWindow}`,
      timestamp: new Date().toISOString(),
      data: { errorCount, timeWindow }
    };

    await this.sendAlert(event);
  }

  async sendDailyLimitAlert(limitType: string, currentValue: number, limitValue: number): Promise<void> {
    const event: AlertEvent = {
      type: 'daily_limit',
      severity: 'warning',
      title: 'Daily Limit Breached',
      message: `${limitType} limit reached: ${currentValue}/${limitValue}`,
      timestamp: new Date().toISOString(),
      data: { limitType, currentValue, limitValue }
    };

    await this.sendAlert(event);
  }

  async sendDrawdownAlert(currentDrawdown: number, limit: number): Promise<void> {
    const event: AlertEvent = {
      type: 'drawdown_breach',
      severity: 'critical',
      title: 'Drawdown Limit Breached',
      message: `Drawdown ${(currentDrawdown * 100).toFixed(2)}% exceeds limit ${(limit * 100).toFixed(2)}%`,
      timestamp: new Date().toISOString(),
      data: { currentDrawdown, limit }
    };

    await this.sendAlert(event);
  }

  async sendHeartbeatAlert(): Promise<void> {
    const event: AlertEvent = {
      type: 'heartbeat',
      severity: 'info',
      title: 'System Heartbeat',
      message: 'Trading bot is running normally',
      timestamp: new Date().toISOString()
    };

    await this.sendAlert(event);
  }

  // Test methods
  async testTelegramAlert(): Promise<{ success: boolean; error?: string }> {
    const event: AlertEvent = {
      type: 'heartbeat',
      severity: 'info',
      title: 'Telegram Test Alert',
      message: 'This is a test message to verify Telegram integration',
      timestamp: new Date().toISOString()
    };

    return await this.sendAlert(event);
  }

  async testEmailAlert(): Promise<{ success: boolean; error?: string }> {
    const event: AlertEvent = {
      type: 'heartbeat',
      severity: 'info',
      title: 'Email Test Alert',
      message: 'This is a test message to verify email integration',
      timestamp: new Date().toISOString()
    };

    return await this.sendAlert(event);
  }

  // Configuration methods
  updateConfig(newConfig: Partial<AlertConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  getConfig(): AlertConfig {
    return { ...this.config };
  }
}
