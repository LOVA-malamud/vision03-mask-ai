export type ModelInfo = {
  version: string;
  test_accuracy: number;
  test_count: number;
  train_count: number;
  validation_count: number;
  validation_accuracy: number;
  best_epoch: number;
};
export type Health = {
  model_ready: boolean;
  model: ModelInfo | null;
};
export type Prediction = {
  prediction: "with_mask" | "without_mask";
  probability_with_mask: number;
  probability_without_mask: number;
  model_version: string;
};
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new Error("לא הצלחנו להתחבר לשרת. ודאו ששרת ה־Python פועל ונסו שוב.");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const messages: Record<number, string> = {
      401: "מפתח הניהול אינו תקין. בדקו את ADMIN_TOKEN בקובץ ההגדרות.",
      403: "הגישה מהכתובת הזו אינה מורשית.",
      409: "התמונה כבר קיימת במאגר, שייכת למבחן, או שאימון כבר מתבצע.",
      413: "התמונה גדולה מדי. בחרו קובץ עד 10 MB.",
      415: "בחרו תמונה בפורמט JPG, PNG או WebP.",
      422: "לא ניתן לקרוא את התמונה או את פרטי הבקשה. נסו קובץ תקין אחר.",
      503: "הפעולה עדיין אינה זמינה. בדקו שהמודל אומן ושקובצי המודל הותקנו.",
      500: "אירעה שגיאה בשרת. נסו שוב ובדקו את יומן השרת.",
    };
    throw new Error(
      messages[response.status] ||
        (typeof data?.detail === "string"
          ? data.detail
          : "הבקשה נכשלה. נסו שוב."),
    );
  }
  return data as T;
}
export function validateFile(file: File): string | null {
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type))
    return "בחרו תמונת JPG, PNG או WebP. אפשר לייצא תמונת HEIC כ־JPEG.";
  if (file.size > 10 * 1024 * 1024)
    return "התמונה גדולה מדי. הגודל המרבי הוא 10 MB.";
  if (!file.size) return "הקובץ ריק. בחרו תמונה אחרת.";
  return null;
}
export const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
