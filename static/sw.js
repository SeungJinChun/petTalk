const CACHE_NAME = "gecko-care-v4";
const APP_SHELL = [
    "/",
    "/login",
    "/main",
    "/survey",
    "/manifest.webmanifest",
    "/static/style.css",
    "/static/supabase_auth.js",
    "/static/icons/icon-192.png",
    "/static/icons/icon-512.png",
    "/static/offline.html",
];

self.addEventListener("install", (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)),
    );
    self.skipWaiting();
});

self.addEventListener("activate", (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys
                    .filter((key) => key !== CACHE_NAME)
                    .map((key) => caches.delete(key)),
            ),
        ),
    );
    self.clients.claim();
});

self.addEventListener("fetch", (event) => {
    const { request } = event;
    if (request.method !== "GET") {
        return;
    }
    const url = new URL(request.url);

    if (request.mode === "navigate") {
        event.respondWith(
            fetch(request).catch(async () => {
                const cached = await caches.match(request, { ignoreSearch: true });
                return cached || caches.match("/static/offline.html");
            }),
        );
        return;
    }

    // Keep critical static assets fresh when online.
    if (url.origin === self.location.origin && url.pathname.startsWith("/static/")) {
        event.respondWith(
            fetch(request)
                .then((response) => {
                    if (response && response.status === 200 && response.type === "basic") {
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME).then((cache) => cache.put(request, responseClone));
                    }
                    return response;
                })
                .catch(async () => {
                    const cached = await caches.match(request, { ignoreSearch: true });
                    return cached || Response.error();
                }),
        );
        return;
    }

    event.respondWith(
        caches.match(request, { ignoreSearch: true }).then((cached) => {
            if (cached) {
                return cached;
            }

            return fetch(request)
                .then((response) => {
                    if (!response || response.status !== 200 || response.type !== "basic") {
                        return response;
                    }

                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(request, responseClone));
                    return response;
                })
                .catch(() => Response.error());
        }),
    );
});
