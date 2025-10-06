import axios from 'axios';

export interface AuthUser {
  id: string;
  username: string;
  role: string;
  permissions: string[];
}

export interface LoginResponse {
  success: boolean;
  token?: string;
  user?: AuthUser;
  message?: string;
}

class AuthService {
  private baseUrl: string;
  private token: string | null = null;
  private user: AuthUser | null = null;

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    this.token = localStorage.getItem('auth_token');
    this.user = this.getStoredUser();
  }

  async login(username: string, password: string): Promise<LoginResponse> {
    try {
      const response = await axios.post(`${this.baseUrl}/auth/login`, {
        username,
        password
      });

      if (response.data.success) {
        this.token = response.data.token;
        this.user = response.data.user;
        
        // Store in localStorage
        localStorage.setItem('auth_token', this.token);
        localStorage.setItem('auth_user', JSON.stringify(this.user));
        
        // Set default authorization header
        axios.defaults.headers.common['Authorization'] = `Bearer ${this.token}`;
      }

      return response.data;
    } catch (error: any) {
      console.error('Login error:', error);
      return {
        success: false,
        message: error.response?.data?.message || 'Login failed'
      };
    }
  }

  async logout(): Promise<void> {
    try {
      if (this.token) {
        await axios.post(`${this.baseUrl}/auth/logout`, {}, {
          headers: { Authorization: `Bearer ${this.token}` }
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      this.clearAuth();
    }
  }

  async refreshToken(): Promise<boolean> {
    try {
      if (!this.token) return false;

      const response = await axios.post(`${this.baseUrl}/auth/refresh`, {}, {
        headers: { Authorization: `Bearer ${this.token}` }
      });

      if (response.data.success) {
        this.token = response.data.token;
        localStorage.setItem('auth_token', this.token);
        axios.defaults.headers.common['Authorization'] = `Bearer ${this.token}`;
        return true;
      }
    } catch (error) {
      console.error('Token refresh error:', error);
    }
    
    return false;
  }

  isAuthenticated(): boolean {
    return !!this.token && !!this.user;
  }

  getToken(): string | null {
    return this.token;
  }

  getUser(): AuthUser | null {
    return this.user;
  }

  hasPermission(permission: string): boolean {
    if (!this.user) return false;
    return this.user.permissions.includes(permission) || this.user.role === 'admin';
  }

  private getStoredUser(): AuthUser | null {
    try {
      const stored = localStorage.getItem('auth_user');
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  }

  private clearAuth(): void {
    this.token = null;
    this.user = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
    delete axios.defaults.headers.common['Authorization'];
  }

  // Initialize auth state on app start
  initialize(): void {
    if (this.token && this.user) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${this.token}`;
    }
  }
}

export const authService = new AuthService();
