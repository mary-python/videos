from manim_imports_ext import *


class WhyDoWeCare(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher

        self.play(
            self.change_students("confused", "erm", "concentrating", look_at=self.screen),
        )
        self.wait(3)
        self.play(
            stds[2].change("erm", stds[1].eyes),
            stds[1].says("I’m sorry, why\ndo we care?", mode="sassy"),
            stds[0].change("thinking", self.screen),
            morty.change("well"),
        )
        self.wait(2)
        self.play(self.change_students("pondering", "maybe", "pondering", look_at=self.screen))

        # Answer
        self.play(
            morty.says("Topology has\nmore subtle utility", mode="tease"),
            stds[0].animate.look_at(morty.eyes),
            stds[1].debubble(),
            stds[2].change("hesitant", morty.eyes)
        )
        self.wait(3)


class RenameTheorem(InteractiveScene):
    def construct(self):
        # Test
        name1, name2 = names = VGroup(
            Text("Hairy Ball Theorem"),
            Text("Sphere Vector Field Theorem"),
        )
        names.scale(1.25)
        names.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        names.to_edge(LEFT)

        lines = VGroup()
        for text in ["Hairy", "Ball"]:
            word = name1[text][0]
            line = Line(word.get_left(), word.get_right())
            line.set_stroke(RED, 8)
            lines.add(line)
        lines[0].align_to(lines[1], UP)

        self.add(name1)
        self.wait()
        self.play(
            ShowCreation(lines[1]),
            name1["Ball"].animate.set_opacity(0.5),
            FadeTransformPieces(name1["Ball"].copy(), name2["Sphere"]),
        )
        self.play(
            ShowCreation(lines[0]),
            name1["Hairy"].animate.set_opacity(0.5),
            FadeTransformPieces(name1["Hairy"].copy(), name2["Vector Field"]),
        )
        self.play(
            TransformFromCopy(name1["Theorem"], name2["Theorem"]),
        )
        self.wait()


class SimpleImplies(InteractiveScene):
    def construct(self):
        arrow = Tex(R"\Rightarrow", font_size=120)
        self.play(Write(arrow))
        self.wait()


class LazyPerpCodeSnippet(InteractiveScene):
    def construct(self):
        # Test
        code = Code("""
            def lazy_perp(heading):
                # Returns the normalized cross product
                # between (0, 0, 1) and heading
                # Note the division by 0 for x=y=0
                x, y, z = heading
                return np.array([-y, x, 0]) / np.sqrt(x * x + y * y)
        """, alignment="LEFT")
        code.to_corner(UL)
        self.play(Write(code))
        self.wait()


class StatementOfTheorem(InteractiveScene):
    def construct(self):
        # Add text
        title = Text("Hairy Ball Theorem", font_size=72)
        title.to_corner(UL)
        underline = Underline(title)

        self.add(title, underline)

        statement = Text("""
            Any continuous vector field
            on a sphere must have at least
            one null vector.
        """, alignment="LEFT")
        statement.next_to(underline, DOWN, buff=MED_LARGE_BUFF)
        statement.to_edge(LEFT)

        self.play(Write(statement, run_time=3, lag_ratio=1e-1))
        self.wait()

        statement.set_backstroke(BLACK, 5)

        # Highlight text
        for text, color in [("continuous", BLUE), ("one null vector", YELLOW)]:
            self.play(
                FlashUnder(statement[text], time_width=1.5, run_time=2, color=color),
                statement[text].animate.set_fill(color)
            )
            self.wait()


class WriteAntipode(InteractiveScene):
    def construct(self):
        # Test
        text1 = Text("“Antipodes”")
        text2 = Text("Antipode map")
        for text in [text1, text2]:
            text.scale(1.5)
            text.to_corner(UL)

        self.play(Write(text1), run_time=2)
        self.wait()
        self.play(TransformMatchingStrings(text1, text2), run_time=1)
        self.wait()


class Programmer(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().fix_in_frame())
        laptop = Laptop()
        self.frame.reorient(60, 66, 0, (0.09, -0.5, 0.13), 4.12)

        randy = Randolph(height=5)
        randy.to_edge(LEFT)
        randy.add_updater(lambda m: m.fix_in_frame().look_at(4 * RIGHT))

        self.add(laptop)
        self.play(randy.change("hesitant"))
        self.play(Blink(randy))
        self.play(randy.change("concentrating"))
        self.play(Blink(randy))
        self.wait()


class ProofOutline(InteractiveScene):
    def construct(self):
        # Add outline
        title = Text("Proof by Contradiction", font_size=72)
        title.to_edge(UP)
        background = FullScreenRectangle()

        frames = Square().replicate(2)
        frames.set_height(4.5)
        frames.arrange(RIGHT, buff=3.5)
        frames.next_to(title, DOWN, buff=1.5)
        frames.set_fill(BLACK, 1)
        frames.set_stroke(WHITE, 2)

        implies = Tex(R"\Longrightarrow", font_size=120)
        implies.move_to(frames)

        impossibility = Text("Impossibility", font_size=90)
        impossibility.next_to(implies, RIGHT, MED_LARGE_BUFF)
        impossibility.set_color(RED)

        assumption = Text("Assume there exists a non-zero\ncontinuous vector field", font_size=30)
        assumption.set_color(BLUE)
        assumption.next_to(frames[0], UP)

        self.add(background)
        self.play(Write(title), run_time=2)
        self.wait()
        self.play(
            FadeIn(frames[0]),
            # FadeIn(assumption, lag_ratio=0.01)
        )
        self.wait()
        implies.save_state()
        implies.stretch(0, 0, about_edge=LEFT)
        self.play(Restore(implies))
        self.play(FadeIn(impossibility, lag_ratio=0.1))
        self.wait()
        self.play(
            DrawBorderThenFill(frames[1]),
            impossibility.animate.scale(0.5).next_to(frames[1], UP)
        )
        self.wait()
        self.play(FadeOut(impossibility))

        # Next part
        words = VGroup(Text("Assume the impossible"), Text("Find a contradiction"))
        brace = Brace(frames[0], RIGHT)
        question = Text("What do we\nshow here?", font_size=72)
        question.next_to(brace, RIGHT)

        self.play(
            FadeOut(implies),
            FadeOut(frames[1]),
        )
        self.play(
            GrowFromCenter(brace),
            FadeIn(question, lag_ratio=0.1),
        )
        self.wait()


class TwoKeyFeatures(InteractiveScene):
    def construct(self):
        # Test
        features = VGroup(
            Text("1) Sphere turns\ninside out"),
            Text("2) No point touches\nthe origin"),
        )
        features[1]["the origin"].align_to(features[1]["No"], LEFT)
        features.arrange(DOWN, aligned_edge=LEFT, buff=1.5)
        features.to_edge(LEFT)

        for feature in features:
            self.play(FadeIn(feature, lag_ratio=0.1))
            self.wait()

        # Emphasize first point
        self.play(
            features[0].animate.scale(1.25, about_edge=LEFT),
            features[1].animate.scale(0.75, about_edge=LEFT).set_fill(opacity=0.5),
        )
        self.wait()

        # Inside out implication
        rect0 = SurroundingRectangle(features[0])
        rect0.set_stroke(BLUE, 2)

        implies0 = Tex(R"\Longrightarrow", font_size=72)
        implies0.next_to(rect0)
        net_flow_m1 = TexText("Net flow ends at $-1.0$", t2c={"-1.0": RED}, font_size=60)
        net_flow_m1.next_to(implies0, RIGHT)

        self.play(
            ShowCreation(rect0),
            Write(implies0),
        )
        self.play(FadeIn(net_flow_m1, lag_ratio=0.1))
        self.wait()

        # No origin implication
        self.play(features[1].animate.scale(1.25 / 0.75, about_edge=UL).set_opacity(1))

        rect1 = SurroundingRectangle(features[1])
        rect1.match_style(rect0)
        implies1 = implies0.copy()
        implies1.next_to(rect1)
        net_flow_p1 = TexText(R"Net flow stays\\constant at $+1.0$", t2c={"+1.0": GREEN}, font_size=60)
        net_flow_p1.next_to(implies1, RIGHT)

        self.play(
            ShowCreation(rect1),
            Write(implies1),
        )
        self.play(FadeIn(net_flow_p1, lag_ratio=0.1))
        self.wait()

        # Contradiction
        contra = Tex(R"\bot", font_size=90)
        contra.to_corner(DR)

        self.play(Write(contra))
        self.wait()


class WhatIsInsideAndOutside(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher

        self.play(
            stds[2].says(
                "Hang on, what\ndo you mean\n“paint the outside”?",
                mode="maybe",
                bubble_direction=LEFT
            ),
            stds[1].change("erm", self.screen),
            stds[0].change("pondering", self.screen),
            morty.change("tease")
        )
        self.wait(5)


class ReferenceInsideOutMovie(TeacherStudentsScene):
    def construct(self):
        # Complain
        morty = self.teacher
        stds = self.students
        self.screen.to_corner(UL)

        self.play(
            stds[0].change("pondering", self.screen),
            stds[1].change("erm", self.screen),
            stds[2].says(
                Text(
                    "Huh? I thought\nyou can turn a\nsphere inside out!",
                    t2s={"can": ITALIC},
                    font_size=42,
                ),
                mode="confused",
                look_at=self.screen,
                bubble_direction=LEFT
            ),
            morty.change("guilty")
        )
        self.wait(2)
        self.play(morty.change('tease'))
        self.wait(3)


class FluxDecimals(InteractiveScene):
    def construct(self):
        # Test
        label = TexText("Flux: +1.000 L/s", font_size=60)
        dec = label.make_number_changeable("+1.000", include_sign=True)
        label.to_corner(UR)
        dec.set_value(0.014)

        def update_color(dec, epsilon=1e-4):
            value = dec.get_value()
            if value > epsilon:
                dec.set_color(GREEN)
            elif abs(value) < epsilon:
                dec.set_color(YELLOW)
            else:
                dec.set_color(RED)

        dec.add_updater(update_color)

        self.add(label)
        self.wait()
        for x in range(2):
            self.play(ChangeDecimalToValue(dec, -dec.get_value()))
            self.wait()
        self.play(ChangeDecimalToValue(dec, 1.0), run_time=3)
        self.wait()
        dec.set_value(0)
        self.wait()


class CommentOnContardiction(InteractiveScene):
    def construct(self):
        # Inside out implies final net flow = -1
        # Never crosses the origin implies
        pass