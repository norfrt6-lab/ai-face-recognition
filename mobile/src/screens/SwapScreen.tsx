import React, { useState } from "react";
import {
  Alert,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import Animated from "react-native-reanimated";
import GradientHeader from "../components/GradientHeader";
import ImagePickerButton from "../components/ImagePickerButton";
import LoadingOverlay from "../components/LoadingOverlay";
import ActionButton from "../components/ActionButton";
import SectionLabel from "../components/SectionLabel";
import { enterPresets } from "../animations";
import { swapFaces } from "../api/client";
import { colors, typography, spacing, radii, shadows } from "../theme";

export default function SwapScreen() {
  const [sourceUri, setSourceUri] = useState<string | null>(null);
  const [targetUri, setTargetUri] = useState<string | null>(null);
  const [resultBase64, setResultBase64] = useState<string | null>(null);
  const [enhance, setEnhance] = useState(false);
  const [loading, setLoading] = useState(false);
  const [timing, setTiming] = useState<number | null>(null);

  const doSwap = async () => {
    if (!sourceUri || !targetUri) {
      return Alert.alert(
        "Missing images",
        "Please select both source and target images.",
      );
    }
    setLoading(true);
    setResultBase64(null);
    setTiming(null);
    try {
      const res = await swapFaces(sourceUri, targetUri, { enhance });
      setResultBase64(res.output_base64);
      setTiming(res.total_inference_ms);
    } catch (e: any) {
      Alert.alert("Swap Failed", e.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setSourceUri(null);
    setTargetUri(null);
    setResultBase64(null);
    setTiming(null);
  };

  return (
    <View style={styles.container}>
      <LoadingOverlay visible={loading} message="Swapping faces..." />
      <ScrollView contentContainerStyle={styles.content}>
        <GradientHeader
          title="Face Swap"
          subtitle="Swap faces between images"
          icon="swap-horizontal"
          compact
        />

        <View style={styles.body}>
          {/* Image Pickers */}
          <Animated.View entering={enterPresets.stagger(0)} style={styles.row}>
            <View style={styles.half}>
              <SectionLabel text="Source Face" />
              <ImagePickerButton
                uri={sourceUri}
                onPick={setSourceUri}
                label="Source"
              />
            </View>
            <View style={styles.arrowWrap}>
              <View style={styles.arrowCircle}>
                <Ionicons
                  name="arrow-forward"
                  size={18}
                  color={colors.primary}
                />
              </View>
            </View>
            <View style={styles.half}>
              <SectionLabel text="Target Scene" />
              <ImagePickerButton
                uri={targetUri}
                onPick={setTargetUri}
                label="Target"
              />
            </View>
          </Animated.View>

          {/* Options */}
          <Animated.View
            entering={enterPresets.stagger(1)}
            style={styles.optionRow}
          >
            <View style={styles.optionInfo}>
              <Text style={styles.optionLabel}>Enhance (GFPGAN)</Text>
              <Text style={styles.optionHint}>Improve face quality</Text>
            </View>
            <Switch
              value={enhance}
              onValueChange={setEnhance}
              trackColor={{
                false: colors.surfaceLight,
                true: colors.primary,
              }}
              thumbColor={colors.text}
            />
          </Animated.View>

          {/* Action */}
          <Animated.View entering={enterPresets.stagger(2)}>
            <ActionButton
              title="Swap Faces"
              icon="swap-horizontal"
              onPress={doSwap}
              disabled={!sourceUri || !targetUri}
              loading={loading}
            />
          </Animated.View>

          {/* Result */}
          {resultBase64 && (
            <Animated.View
              entering={enterPresets.fadeInUp}
              style={[styles.resultCard, shadows.sm]}
            >
              <View style={styles.resultHeader}>
                <Ionicons
                  name="checkmark-circle"
                  size={20}
                  color={colors.success}
                />
                <Text style={styles.resultTitle}>Swap Complete</Text>
                {timing != null && (
                  <View style={styles.timingPill}>
                    <Text style={styles.timingText}>
                      {timing.toFixed(0)}ms
                    </Text>
                  </View>
                )}
              </View>
              <Image
                source={{ uri: `data:image/png;base64,${resultBase64}` }}
                style={styles.resultImage}
                resizeMode="contain"
              />
            </Animated.View>
          )}

          {/* Reset */}
          {(sourceUri || targetUri || resultBase64) && (
            <Pressable style={styles.resetBtn} onPress={reset}>
              <Ionicons
                name="refresh-outline"
                size={16}
                color={colors.textSecondary}
              />
              <Text style={styles.resetText}>Reset</Text>
            </Pressable>
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
  row: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: spacing.sm,
  },
  half: { flex: 1 },
  arrowWrap: {
    paddingBottom: spacing.xxxl,
  },
  arrowCircle: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.primaryMuted,
    alignItems: "center",
    justifyContent: "center",
  },
  optionRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  optionInfo: { flex: 1 },
  optionLabel: {
    ...typography.body,
    color: colors.text,
  },
  optionHint: {
    ...typography.caption,
    color: colors.textMuted,
    marginTop: 2,
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
    marginBottom: spacing.md,
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
  resultImage: {
    width: "100%",
    aspectRatio: 1,
    borderRadius: radii.md,
  },
  resetBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.sm,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
  },
  resetText: {
    ...typography.bodySmall,
    color: colors.textSecondary,
  },
});
