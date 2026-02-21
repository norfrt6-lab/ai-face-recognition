export interface ComponentHealth {
  status: "ok" | "degraded" | "down";
  loaded: boolean;
  detail: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "down";
  version: string;
  environment: string;
  uptime_seconds: number;
  components: Record<string, ComponentHealth>;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

export interface LandmarkPoint {
  x: number;
  y: number;
}

export interface FaceMatch {
  identity_name: string;
  identity_id: string;
  similarity: number;
  is_known: boolean;
  threshold_used: number;
}

export interface FaceAttributes {
  age: number;
  gender: string;
  gender_score: number;
}

export interface DetectedFace {
  face_index: number;
  bbox: BoundingBox;
  landmarks?: LandmarkPoint[];
  attributes?: FaceAttributes;
  match?: FaceMatch;
}

export interface RecognizeResponse {
  num_faces_detected: number;
  num_faces_recognized: number;
  faces: DetectedFace[];
  inference_time_ms: number;
  image_width: number;
  image_height: number;
}

export interface SwapTimingBreakdown {
  align_ms: number;
  inference_ms: number;
  blend_ms: number;
  total_ms: number;
}

export interface SwappedFaceInfo {
  face_index: number;
  bbox: BoundingBox;
  success: boolean;
  status: string;
  timing: SwapTimingBreakdown;
  error: string | null;
}

export interface SwapResponse {
  output_url: string | null;
  output_base64: string;
  num_faces_swapped: number;
  num_faces_failed: number;
  faces: SwappedFaceInfo[];
  total_inference_ms: number;
  blend_mode: string;
  enhanced: boolean;
  watermarked: boolean;
}

export interface RegisterResponse {
  identity_id: string;
  identity_name: string;
  embeddings_added: number;
  total_embeddings: number;
  faces_detected: number;
  message: string;
}

export interface IdentityItem {
  name: string;
  identity_id: string;
  num_embeddings: number;
  created_at?: string;
  updated_at?: string;
}

export interface IdentitiesListResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  items: IdentityItem[];
}
