# TheraGraph frontend image: build the static bundle, serve it from nginx.

FROM node:26-alpine AS builder

WORKDIR /build

# Lockfile-only install first so the dependency layer caches. `npm ci` and nothing else:
# falling back to `npm install` would silently build from versions the lock never saw.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
# Baked in at build time: the bundle is static, so the API base must be known now.
ARG VITE_API_BASE=/api/v1
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build

# The unprivileged variant runs nginx as uid 101 and listens on 8080, so the container
# never holds root.
FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime

USER root
RUN rm -f /etc/nginx/conf.d/default.conf && mkdir -p /etc/nginx/snippets
COPY deploy/nginx.conf /etc/nginx/conf.d/theragraph.conf
COPY deploy/security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY --from=builder /build/dist /usr/share/nginx/html
USER 101

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -q --spider http://localhost:8080/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
