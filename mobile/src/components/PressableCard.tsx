import React from "react";
import { Pressable, StyleSheet, ViewStyle } from "react-native";
import Animated from "react-native-reanimated";
import { useScalePress } from "../animations";
import { colors, radii, shadows, spacing } from "../theme";

interface Props {
  children: React.ReactNode;
  onPress?: () => void;
  style?: ViewStyle;
  entering?: any;
}

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

export default function PressableCard({ children, onPress, style, entering }: Props) {
  const { animatedStyle, onPressIn, onPressOut } = useScalePress();

  const card = (
    <AnimatedPressable
      onPress={onPress}
      onPressIn={onPressIn}
      onPressOut={onPressOut}
      style={[styles.card, shadows.sm, animatedStyle, style]}
    >
      {children}
    </AnimatedPressable>
  );

  if (entering) {
    return <Animated.View entering={entering}>{card}</Animated.View>;
  }

  return card;
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
});
