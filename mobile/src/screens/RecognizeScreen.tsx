import React, { useState } from "react";
import { Alert, ScrollView, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import Animated from "react-native-reanimated";
import GradientHeader from "../components/GradientHeader";
import ImagePickerButton from "../components/ImagePickerButton";
import LoadingOverlay from "../components/LoadingOverlay";
import ActionButton from "../components/ActionButton";
import SectionLabel from "../components/SectionLabel";
import { enterPresets } from "../animations";
import { recognizeFaces } from "../api/client";
import type { RecognizeResponse } from "../types/api";
import { colors, typography, spacing, radii, shadows } from "../theme";

export default function RecognizeScreen() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<RecognizeResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const doRecognize = async () => {
    if (!imageUri)
      return Alert.alert("No image", "Please select an image first.");
    setLoading(true);
    setResult(null);
    try {
      const res = await recognizeFaces(imageUri);
      setResult(res);
    } catch (e: any) {
      Alert.alert("Recognition Failed", e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <LoadingOverlay visible={loading} message="Analyzing faces..." />
      <ScrollView contentContainerStyle={styles.content}>
        <GradientHeader
          title="Recognize"
          subtitle="Detect and identify faces"
          icon="scan"
          compact
        />

        <View style={styles.body}>
          <Animated.View entering={enterPresets.stagger(0)}>
            <SectionLabel text="Select an Image" />
            <View style={styles.pickerWrap}>
              <ImagePickerButton
                uri={imageUri}
                onPick={setImageUri}
                label="Select Image"
              />
            </View>
          </Animated.View>

          <Animated.View entering={enterPresets.stagger(1)}>
            <ActionButton
              title="Recognize Faces"
              icon="scan"
              onPress={doRecognize}
              disabled={!imageUri}
              loading={loading}
            />
          </Animated.View>

          {result && (
            <Animated.View
              entering={enterPresets.fadeInUp}
              style={[styles.resultCard, shadows.sm]}
            >
              <View style={styles.resultHeader}>
                <Ionicons
                  name="people-circle-outline"
                  size={22}
                  color={colors.primary}
                />
                <Text style={styles.resultTitle}>
                  {result.num_faces_detected} face(s) detected
                </Text>
                <View style={styles.timingPill}>
                  <Text style={styles.timingText}>
                    {result.inference_time_ms.toFixed(0)}ms
                  </Text>
                </View>
              </View>

              {result.faces.map((face, index) => (
                <Animated.View
                  key={face.face_index}
                  entering={enterPresets.stagger(index)}
                  style={styles.faceCard}
                >
                  <View style={styles.faceHeader}>
                    <Text style={styles.faceTitle}>
                      Face #{face.face_index + 1}
                    </Text>
                    <View style={styles.confidencePill}>
                      <Text style={styles.confidenceText}>
                        {(face.bbox.confidence * 100).toFixed(1)}%
                      </Text>
                    </View>
                  </View>

                  {face.attributes && (
                    <View style={styles.attrRow}>
                      <View style={styles.attrPill}>
                        <Ionicons
                          name="calendar-outline"
                          size={12}
                          color={colors.textSecondary}
                        />
                        <Text style={styles.attrText}>
                          ~{face.attributes.age.toFixed(0)} yrs
                        </Text>
                      </View>
                      <View style={styles.attrPill}>
                        <Ionicons
                          name="person-outline"
                          size={12}
                          color={colors.textSecondary}
                        />
                        <Text style={styles.attrText}>
                          {face.attributes.gender} (
                          {(face.attributes.gender_score * 100).toFixed(0)}%)
                        </Text>
                      </View>
                    </View>
                  )}

                  {face.match?.is_known ? (
                    <View style={styles.matchBox}>
                      <View style={styles.matchInfo}>
                        <Ionicons
                          name="checkmark-circle"
                          size={18}
                          color={colors.success}
                        />
                        <Text style={styles.matchName}>
                          {face.match.identity_name}
                        </Text>
                      </View>
                      <Text style={styles.matchScore}>
                        {(face.match.similarity * 100).toFixed(1)}%
                      </Text>
                    </View>
                  ) : (
                    <View style={styles.unknownBox}>
                      <Ionicons
                        name="help-circle-outline"
                        size={16}
                        color={colors.textMuted}
                      />
                      <Text style={styles.unknownText}>Unknown</Text>
                    </View>
                  )}
                </Animated.View>
              ))}
            </Animated.View>
          )}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { paddingBottom: 100 },
  body: {
    paddingHorizontal: spacing.xl,
    marginTop: spacing.lg,
    gap: spacing.lg,
  },
  pickerWrap: {
    width: "70%",
    alignSelf: "center",
  },
  resultCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  resultHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  resultTitle: {
    ...typography.h3,
    color: colors.text,
    flex: 1,
  },
  timingPill: {
    backgroundColor: colors.surfaceLight,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
  },
  timingText: {
    ...typography.caption,
    color: colors.textMuted,
  },
  faceCard: {
    backgroundColor: colors.surfaceLight,
    borderRadius: radii.md,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderLeftWidth: 3,
    borderLeftColor: colors.primary,
  },
  faceHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: spacing.sm,
  },
  faceTitle: {
    ...typography.h3,
    color: colors.text,
  },
  confidencePill: {
    backgroundColor: colors.primaryMuted,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: radii.full,
  },
  confidenceText: {
    ...typography.caption,
    color: colors.primary,
    fontWeight: "600",
  },
  attrRow: {
    flexDirection: "row",
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  attrPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    backgroundColor: colors.surface,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
  },
  attrText: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  matchBox: {
    backgroundColor: colors.successMuted,
    borderRadius: radii.sm,
    padding: spacing.md,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  matchInfo: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
  },
  matchName: {
    ...typography.body,
    color: colors.success,
    fontWeight: "600",
  },
  matchScore: {
    ...typography.bodySmall,
    color: colors.success,
  },
  unknownBox: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    paddingTop: spacing.xs,
  },
  unknownText: {
    ...typography.bodySmall,
    color: colors.textMuted,
    fontStyle: "italic",
  },
});
