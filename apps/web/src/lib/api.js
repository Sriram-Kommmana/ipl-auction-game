import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_SERVER_URL,
  timeout: 10000
})

// Unwrap ApiResponse/ApiError shape into a plain Error so callers can just do
// try { await createRoom(...) } catch (err) { toast(err.message) }
api.interceptors.response.use(
  (response) => response.data, // unwrap axios response -> { statusCode, data, message, success }
  (error) => {
    if (error.response?.data?.message) {
      return Promise.reject(new Error(error.response.data.message))
    }
    if (error.request) {
      return Promise.reject(new Error('Could not reach the server. Check your connection.'))
    }
    return Promise.reject(error)
  }
)

export const createRoom = (payload) => api.post('/room/create', payload)
export const joinRoom = (payload) => api.post('/room/join', payload)
export const getRoomState = (roomId) => api.get(`/room/${roomId}`)

export default api