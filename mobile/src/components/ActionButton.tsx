import React from "react";
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import Animated from "react-native-reanimated";
import { useScalePress } from "../animations";
import { colors, typography, radii, shadows, spacing } from "../theme";

interface Props {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  loading?: boolean;
  icon?: keyof typeof Ionicons.glyphMap;
}

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

export default function ActionButton({
  title,
  onPress,
  disabled,
  loading,
  icon,
}: Props) {
  const { animatedStyle, onPressIn, onPressOut } = useScalePress();
  const isDisabled = disabled || loading;

  return (
    <AnimatedPressable
      onPress={onPress}
      onPressIn={isDisabled ? undefined : onPressIn}
      onPressOut={isDisabled ? undefined : onPressOut}
      disabled={isDisabled}
      style={[animatedStyle, isDisabled && styles.disabled]}
    >
      <LinearGradient
        colors={
          isDisabled
            ? [colors.surfaceLight, colors.surface]
            : [colors.primary, colors.primaryDark]
        }
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={[styles.gradient, !isDisabled && shadows.md]}
      >
        {loading ? (
          <ActivityIndicator color={colors.text} size="small" />
        ) : (
          <View style={styles.content}>
            {icon && (
              <Ionicons
                name={icon}
                size={20}
                color={isDisabled ? colors.textMuted : colors.text}
                style={styles.icon}
              />
            )}
            <Text
              style={[
                styles.text,
                isDisabled && { color: colors.textMuted },
              ]}
            >
              {title}
            </Text>
          </View>
        )}
      </LinearGradient>
    </AnimatedPressable>
  );
}

const styles = StyleSheet.create({
  gradient: {
    borderRadius: radii.md,
    paddingVertical: spacing.lg,
    paddingHorizontal: spacing.xl,
    alignItems: "center",
    justifyContent: "center",
  },
  disabled: {
    opacity: 0.5,
  },
  content: {
    flexDirection: "row",
    alignItems: "center",
  },
  icon: {
    marginRight: spacing.sm,
  },
  text: {
    ...typography.h3,
    color: colors.text,
  },
});
