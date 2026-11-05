# Safe page fetching

Web search results come from an external provider, and each result page is
downloaded by the server. To stop a poisoned result from steering that request at
internal services, every URL is checked before it is fetched: only http and https
are allowed, and every address the hostname resolves to must be a public address.
Loopback, private network ranges and the cloud metadata address are refused.

Redirects are not followed, because an allowed public page could otherwise
redirect the fetch to an internal address after the check has passed.
