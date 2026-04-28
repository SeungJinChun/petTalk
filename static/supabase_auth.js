(function () {
  if (window.__supabaseAuthFetchPatched) {
    return;
  }

  window.__supabaseAuthFetchPatched = true;

  const tokenKey = "supabase_access_token";
  const originalFetch = window.fetch.bind(window);

  function getToken() {
    return localStorage.getItem(tokenKey);
  }

  function syncTokenCookie(token) {
    if (!token) {
      document.cookie = `${tokenKey}=; Path=/; SameSite=Lax; Max-Age=0`;
      return;
    }

    document.cookie = `${tokenKey}=${encodeURIComponent(token)}; Path=/; SameSite=Lax; Max-Age=604800`;
  }

  function setToken(token) {
    if (!token) {
      localStorage.removeItem(tokenKey);
      syncTokenCookie(null);
      return;
    }

    localStorage.setItem(tokenKey, token);
    syncTokenCookie(token);
  }

  syncTokenCookie(getToken());

  window.fetch = function (input, init) {
    const requestUrl = typeof input === "string" ? input : input && input.url;
    const url = requestUrl ? new URL(requestUrl, window.location.origin) : null;
    const isApi = url && url.origin === window.location.origin && url.pathname.startsWith("/api/");
    const token = getToken();

    if (isApi && token) {
      init = init || {};
      const headers = new Headers(init.headers || (input instanceof Request ? input.headers : undefined));
      headers.set("Authorization", `Bearer ${token}`);

      init.headers = {
        ...Object.fromEntries(headers.entries()),
      };
    }

    return originalFetch(input, init);
  };

  window.auth = {
    getToken() {
      return getToken();
    },
    setToken,
    me() {
      return fetch("/api/auth/me").then((response) => response.json());
    },
    logout() {
      setToken(null);
      location.href = "/login";
    },
  };
})();
