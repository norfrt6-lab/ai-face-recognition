import { useEffect } from "react";
import {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSpring,
  withDelay,
  withRepeat,
  Easing,
  FadeIn,
  FadeInDown,
  FadeInUp,
  FadeOut,
} from "react-native-reanimated";

export const enterPresets = {
  fadeIn: FadeIn.duration(400),
  fadeInDown: FadeInDown.duration(500).springify().damping(15),
  fadeInUp: FadeInUp.duration(400),
  fadeOut: FadeOut.duration(200),
  stagger: (index: number) =>
    FadeInDown.delay(index * 80)
      .duration(400)
      .springify()
      .damping(15),
};

export function useScalePress() {
  const scale = useSharedValue(1);
  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));
  const onPressIn = () => {
    scale.value = withSpring(0.96, { damping: 15, stiffness: 300 });
  };
  const onPressOut = () => {
    scale.value = withSpring(1, { damping: 15, stiffness: 300 });
  };
  return { animatedStyle, onPressIn, onPressOut };
}

export function usePulse(active: boolean) {
  const opacity = useSharedValue(1);
  useEffect(() => {
    if (active) {
      opacity.value = withRepeat(
        withTiming(0.3, {
          duration: 1000,
          easing: Easing.inOut(Easing.ease),
        }),
        -1,
        true,
      );
    } else {
      opacity.value = withTiming(1, { duration: 300 });
    }
  }, [active]);
  const animatedStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));
  return animatedStyle;
}

export function useFadeIn(delay = 0) {
  const opacity = useSharedValue(0);
  const translateY = useSharedValue(12);
  useEffect(() => {
    opacity.value = withDelay(delay, withTiming(1, { duration: 500 }));
    translateY.value = withDelay(
      delay,
      withSpring(0, { damping: 15 }),
    );
  }, []);
  const animatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ translateY: translateY.value }],
  }));
  return animatedStyle;
}
