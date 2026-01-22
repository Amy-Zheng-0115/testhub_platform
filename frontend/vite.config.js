import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

// 抑制 Node.js util._extend 弃用警告（由依赖包使用旧 API 引起）
// 这个警告来自某些旧版本的 npm 包，不影响功能
if (typeof process !== 'undefined') {
  const originalEmitWarning = process.emitWarning
  process.emitWarning = function(warning, ...args) {
    if (warning && typeof warning === 'object' && warning.name === 'DeprecationWarning') {
      if (warning.message && warning.message.includes('util._extend')) {
        // 静默处理 util._extend 弃用警告
        return
      }
    } else if (typeof warning === 'string' && warning.includes('util._extend')) {
      // 静默处理字符串格式的警告
      return
    }
    // 其他警告正常显示
    return originalEmitWarning.call(process, warning, ...args)
  }
}

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  css: {
    preprocessorOptions: {
      scss: {
        // 静默弃用警告，将在 Sass 2.0.0 前解决
        silenceDeprecations: ['legacy-js-api'],
      },
      sass: {
        // 静默弃用警告，将在 Sass 2.0.0 前解决
        silenceDeprecations: ['legacy-js-api'],
      },
    },
  },
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '^/api/': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        timeout: 30000,  // 增加超时时间到30秒
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.error('代理错误:', err.message);
            console.error('请确保 Django 后端服务器正在运行在 http://127.0.0.1:8000');
            console.error('启动命令: python manage.py runserver');
          });
        },
      },
      '^/media/': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        timeout: 30000,
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.error('媒体文件代理错误:', err.message);
            console.error('请确保 Django 后端服务器正在运行在 http://127.0.0.1:8000');
          });
        },
      },
    },
    historyApiFallback: {
      index: '/index.html',
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
  },
  optimizeDeps: {
    include: ['monaco-editor']
  }
})