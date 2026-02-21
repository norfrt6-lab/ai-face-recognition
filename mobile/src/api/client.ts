import { API_BASE_URL } from "../constants";
import type {
  HealthResponse,
  IdentitiesListResponse,
  RecognizeResponse,
  RegisterResponse,
  SwapResponse,
} from "../types/api";

function imageFile(uri: string) {
  const name = uri.split("/").pop() || "image.jpg";
  const ext = name.split(".").pop()?.toLowerCase() || "jpg";
  const type = ext === "png" ? "image/png" : "image/jpeg";
  return { uri, name, type } as any;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function swapFaces(
  sourceUri: string,
  targetUri: string,
  options?: { blendMode?: string; enhance?: boolean }
): Promise<SwapResponse> {
  const form = new FormData();
  form.append("source_file", imageFile(sourceUri));
  form.append("target_file", imageFile(targetUri));
  form.append("consent", "true");
  form.append("watermark", "false");
  form.append("return_base64", "true");
  form.append("blend_mode", options?.blendMode || "poisson");
  form.append("enhance", String(options?.enhance ?? false));

  const res = await fetch(`${API_BASE_URL}/swap`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.message || `Swap failed: ${res.status}`);
  }
  return res.json();
}

export async function recognizeFaces(
  imageUri: string
): Promise<RecognizeResponse> {
  const form = new FormData();
  form.append("image", imageFile(imageUri));
  form.append("consent", "true");
  form.append("return_attributes", "true");

  const res = await fetch(`${API_BASE_URL}/recognize`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      err.detail || err.message || `Recognition failed: ${res.status}`
    );
  }
  return res.json();
}

export async function registerFace(
  imageUri: string,
  name: string
): Promise<RegisterResponse> {
  const form = new FormData();
  form.append("image", imageFile(imageUri));
  form.append("name", name);
  form.append("consent", "true");

  const res = await fetch(`${API_BASE_URL}/register`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      err.detail || err.message || `Registration failed: ${res.status}`
    );
  }
  return res.json();
}

export async function listIdentities(): Promise<IdentitiesListResponse> {
  const res = await fetch(`${API_BASE_URL}/identities`);
  if (!res.ok) throw new Error(`Failed to list identities: ${res.status}`);
  return res.json();
}

export async function deleteIdentity(identityId: string): Promise<void> {
  const form = new FormData();
  form.append("confirm", "true");

  const res = await fetch(`${API_BASE_URL}/identities/${identityId}`, {
    method: "DELETE",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      err.detail || err.message || `Delete failed: ${res.status}`
    );
  }
}
